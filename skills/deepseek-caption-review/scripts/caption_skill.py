#!/usr/bin/env python3
"""Single-Caption workflow. Python 3.10+ standard library; zero network calls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

import _schema as schema
import _rules as rules
from _audio import audio_spans
import caption_qa_width as width

VERSION='1.0.0'
SEMANTIC_LABELS={'疑似病句','疑似错别字','标点问题','主体引用','内容符号'}
LABELS={'duplicate_defined_subjects':'主体重复定义','missing_subjects':'自然指令漏写已定义主体',
        'ghost_subjects':'引用未定义主体','skipped_numbering':'主体编号不连续',
        'symbol_format':'内容符号不成对','label_separator':'多标签拼接符不规范',
        'bare_tags':'编号标签待核对','missing_punctuation':'段尾标点待核对'}


def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_text(path):
    if path=='-':return sys.stdin.read()
    p=Path(path)
    if p.suffix.lower() not in {'.txt','.md'}:raise ValueError('只接收单段Caption的UTF-8 TXT/MD；不处理表格或批量文件。')
    with p.open(encoding='utf-8-sig',newline='') as stream:return stream.read()


def json_file(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('JSON键重复：'+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)


def sections(text):
    parsed=schema.parse(text);headers=parsed['headers']
    if parsed['unknown'] or any(b['field'] is None for b in parsed['blocks']):raise ValueError('有未知标题或字段外正文，保留原文待核对。')
    names=[h['field'] for h in headers]
    if not headers or len(names)!=len(set(names)):raise ValueError('没有可靠字段标题或存在重复字段，不能自动覆盖。')
    result={}
    for n,h in enumerate(headers):
        if h['inline'] is not None:raise ValueError('标题与正文同一行，请先按字段规范步骤整理。')
        end=headers[n+1]['start'] if n+1<len(headers) else len(text)
        result[h['field']]={'start':h['end'],'end':end,'text':text[h['end']:end]}
    return result


def protected(text):
    spans=list(audio_spans(text))
    spans += [(m.start(),m.end()) for m in width.SCREEN.finditer(text)]
    return sorted(set(spans))


def outside_transform(text,transform):
    spans=protected(text)+[(m.start(),m.end()) for m in width.TECH.finditer(text)]
    spans += [(m.start(),m.end()) for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」',text)]
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    parts=[];cursor=0
    for a,b in merged:parts.extend((transform(text[cursor:a]),text[a:b]));cursor=b
    parts.append(transform(text[cursor:]));return ''.join(parts)


def safe_body(name,text):
    if name in rules.PUNCTUATION_EXEMPT_SECTIONS:return text
    spans=protected(text)
    def allowed(op):
        for a,b in spans:
            if op['start']<b and op['end']>a:
                outer=op['width_rule'] in {'人声包裹符','背景音乐外层括号','音效包裹符'} and (op['start']==a or op['end']==b)
                if not outer:return False
        return True
    edits=[op for op in width.operations(name,text) if allowed(op)]
    for op in reversed(edits):text=text[:op['start']]+op['new']+text[op['end']:]
    if name in rules.LLM_TARGET_SECTIONS:text=outside_transform(text,rules._normalize_markup_tags)
    def typos(value):
        for old,new in rules.UNAMBIGUOUS_TYPO_FIXES.items():value=value.replace(old,new)
        return value
    text=outside_transform(text,typos)
    spans=protected(text);lines=[];offset=0
    for line in text.splitlines(keepends=True):
        original_length=len(line)
        trimmed=line.rstrip();end=offset+len(trimmed)
        inside=any(a<end<b for a,b in spans)
        unclosed=any(end==b and text[a:a+1] in '{｛【<＜（(' and
            (text[a:a+1],text[b-1:b]) not in {('{','}'),('｛','｝'),('【','】'),('<','>'),('＜','＞'),('（','）'),('(',')')} for a,b in spans)
        if trimmed.strip() and trimmed[-1] not in rules.SENTENCE_ENDINGS and not inside and not unclosed:
            line=trimmed+'。'+line[len(trimmed):]
        lines.append(line);offset+=original_length
    return ''.join(lines)


def excerpt(text,snippet):
    if not snippet or snippet not in text:return ''
    pos=text.index(snippet);start=pos;end=pos+len(snippet)
    # Full source line includes its surrounding sentence, avoiding cuts inside audio wrappers.
    start=text.rfind('\n',0,start)+1
    found=text.find('\n',end);end=len(text) if found<0 else found
    return text[start:end]


def finding(field,label,reason,original='',replacement=None):
    return {'field':field,'label':label,'reason':reason,'original':original,
            'replacement':replacement,'decision':'pending'}


def bind_findings(report):
    try:values=sections(report['caption'])
    except ValueError:values={}
    seen=set();out=[];used=set()
    approved_ids={v['issue_id'] for v in report.get('approved_history',[])}
    existing_ids=[v.get('issue_id','') for v in report['findings']]+list(approved_ids)
    next_number=max([report.get('next_issue_number',1)]+[int(v[1:])+1 for v in existing_ids if re.fullmatch(r'Q\d{3,}',v)])
    for item in report['findings']:
        key=(item['field'],item['label'],item.get('original',''),item['reason'])
        if key in seen:continue
        seen.add(key);item=copy.deepcopy(item)
        identifier=item.get('issue_id')
        if not isinstance(identifier,str) or not re.fullmatch(r'Q\d{3,}',identifier) or identifier in used or identifier in approved_ids:
            identifier='Q'+str(next_number).zfill(3);next_number+=1
        item['issue_id']=identifier;used.add(identifier)
        body=values.get(item['field'],{}).get('text','')
        item['excerpt']=excerpt(body,item.get('original',''))
        if any(word in item['label'] for word in ('主体','编号')):item['subject']=values.get('主体',{}).get('text','')
        out.append(item)
    report['findings']=out
    report['next_issue_number']=next_number


def check(caption,sample_id=None,check_only=False):
    if not isinstance(caption,str) or not caption.strip():raise ValueError('Caption不能为空。')
    if len(caption)>200000:raise ValueError('单段Caption超过200000字符，请确认输入范围。')
    report={'version':VERSION,'field_schema_version':schema.VERSION,'sample_id':sample_id,
        'source_caption':caption,'source_sha256':digest(caption),'caption':caption,
        'base_sha256':digest(caption),'mode':'check_only' if check_only else 'safe_fix',
        'semantic_status':'not_run','findings':[],'automatic_changes':[],'approved_history':[],
        'missing_fields':[],'status':'LOCAL_CHECK_COMPLETE'}
    parsed=schema.parse(caption);hs=parsed['headers'];present={h['field'] for h in hs}
    if '_task' in present:present.add('任务类型')
    report['missing_fields']=[f for f in schema.FIELDS if f not in present] if hs else []
    for name in report['missing_fields']:
        report['findings'].append(finding(name,'缺少字段','原始数据缺少字段标题；没有原文依据时仅补空标题，内容须人工核对。'))
    legacy=any(h['alias'] in {'task_type','subtask_type','natural_instruction'} for h in hs)
    if legacy or not hs or parsed['unknown'] or any(b['field'] is None for b in parsed['blocks']) or len(present)!=len(hs):
        report['status']='STRUCTURE_NEEDS_CONFIRMATION'
        report['findings'].append(finding('字段结构','结构待确认','旧英文整套结构、重复字段或未知边界需先确认，原文保留；旧task_type为参考时不要猜新版标签。'))
        bind_findings(report);return report
    prepared,_=schema.local_caption(caption,parsed,schema.task_value(parsed))
    raw_sections=sections(prepared)
    edits=[]
    for name,item in raw_sections.items():
        revised=safe_body(name,item['text'])
        if revised!=item['text']:edits.append((item['start'],item['end'],revised))
    candidate=prepared
    for a,b,value in reversed(edits):candidate=candidate[:a]+value+candidate[b:]
    if candidate!=caption:
        changes=[{'old':caption[a:b],'new':candidate[c:d]} for tag,a,b,c,d in difflib.SequenceMatcher(None,caption,candidate,autojunk=False).get_opcodes() if tag!='equal']
        report['proposed_safe_changes' if check_only else 'automatic_changes']=changes
    report['caption']=caption if check_only else candidate
    if check_only and candidate!=caption:report['findings'].append(finding('格式','可安全修正','当前为只检查模式；明确的格式问题尚未应用，可请求安全修正。'))
    values={k:v['text'] for k,v in sections(candidate).items()}
    for item in rules.run_static_rules(values):
        name=item['section'];snippet=item['original_snippet'];text=values.get(name,'')
        if item['issue_type']=='bare_tags' and snippet in text and all(any(a<=m.start() and m.end()<=b for a,b in protected(text)) for m in re.finditer(re.escape(snippet),text)):continue
        report['findings'].append(finding(name,LABELS.get(item['issue_type'],item['issue_type']),item['suggestion'],snippet if snippet in text else ''))
    for name,item in sections(candidate).items():
        if not item['text'].strip():report['findings'].append(finding(name,'字段内容为空','该字段没有正文；留空不代表已确认“无”，请核对。'))
    report['base_sha256']=digest(report['caption']);bind_findings(report)
    if report['findings']:report['status']='NEEDS_CONFIRMATION'
    return report


def read_report(path):
    report=json_file(path)
    if report.get('version')!=VERSION or report.get('field_schema_version')!=schema.VERSION:raise ValueError('报告规则版本不匹配，请重新检查。')
    if digest(report['source_caption'])!=report.get('source_sha256') or digest(report['caption'])!=report.get('base_sha256'):raise ValueError('原文或当前稿已变化，请重新检查，不能套用旧报告。')
    return report


def candidate_operation(caption,item):
    values=sections(caption);field=item['field']
    if field not in values:raise ValueError('候选字段不在当前Caption中。')
    text=values[field]['text'];old=item.get('original');new=item.get('replacement')
    if not isinstance(old,str) or not old or not isinstance(new,str):raise ValueError('缺少可执行的原文或具体改稿。')
    positions=[m.start() for m in re.finditer(re.escape(old),text)]
    occurrence=item.get('occurrence')
    if occurrence is None:
        if len(positions)!=1:raise ValueError('原文定位不存在或有多处，须指定从1开始的occurrence。')
        pos=positions[0]
    else:
        if type(occurrence) is not int or not 1<=occurrence<=len(positions):raise ValueError('occurrence定位无效。')
        pos=positions[occurrence-1]
    after=text[:pos]+new+text[pos+len(old):]
    if field in rules.PUNCTUATION_EXEMPT_SECTIONS:
        labels=lambda value:re.sub(r'[|｜\s、，,；;]','',value)
        if labels(text)!=labels(after):raise ValueError('元数据仅可确认分隔符格式，不可猜改任务或视听标签含义。')
    if [m[0] for m in width.SCREEN.finditer(text)]!=[m[0] for m in width.SCREEN.finditer(after)]:raise ValueError('画面文字正文必须保留，不能由建议修改。')
    old_audio=[text[a:b] for a,b in audio_spans(text)]
    new_audio=[after[a:b] for a,b in audio_spans(after)]
    audio_changed=old_audio!=new_audio
    if audio_changed:
        # Only exact punctuation-only interior audio proposals can be explicitly approved.
        strip=lambda v:re.sub(r'[，。！？、；：,.!?;:]', '',v)
        if len(old_audio)!=len(new_audio) or [strip(v) for v in old_audio]!=[strip(v) for v in new_audio]:
            raise ValueError('候选改动涉及音频文字、空白或包裹符，需依据素材人工改稿。')
    # Values, timecodes and material/scene references remain fixed after confirmation.
    tag_re=re.compile(r'<(?:主体|场景|图片|视频|音频|画面元素)\d+>')
    tags_before=tag_re.findall(text);tags_after=tag_re.findall(after)
    if item['label']=='主体引用' and field in {'总结','自然指令'}:
        non_subject=lambda tags:[v for v in tags if not v.startswith('<主体')]
        if non_subject(tags_before)!=non_subject(tags_after):raise ValueError('场景与素材编号不可随建议变更。')
        defined=set(rules._subject_definitions(values.get('主体',{}).get('text','')))
        added=Counter(tags_after)-Counter(tags_before)
        if any(v not in defined for v in added):raise ValueError('主体引用建议引入了未定义主体。')
    elif tags_before!=tags_after:raise ValueError('候选改动改变了编号；主体引用须明确使用主体引用标签并人工确认。')
    numbers=lambda value:re.findall(r'\d+(?:[.:/-]\d+)*',tag_re.sub('',value))
    if numbers(text)!=numbers(after):raise ValueError('数值或时间不可随建议变更。')
    if width.TECH.findall(text)!=width.TECH.findall(after):raise ValueError('网址、文件路径或技术数值不可随建议变更。')
    proposed=caption[:values[field]['start']]+after+caption[values[field]['end']:]
    if list(sections(proposed))!=list(values):raise ValueError('候选改动改变了字段标题或边界。')
    return {'start':pos,'end':pos+len(old),'old':old,'new':new,'audio_confirmation_required':audio_changed}


def propose(report,proposal):
    if report['mode']=='check_only':raise ValueError('请先执行安全修正，再基于当前稿提出建议。')
    if proposal.get('base_sha256')!=report['base_sha256']:raise ValueError('建议对应的Caption版本不一致。')
    if proposal.get('semantic_status') not in {'completed','incomplete'}:raise ValueError('必须明确语义检查是否完成。')
    items=proposal.get('findings')
    if not isinstance(items,list):raise ValueError('findings必须是数组。')
    result=copy.deepcopy(report)
    # Replacing an unfinished semantic pass must not silently dismiss pending suggestions.
    existing={ (v['field'],v['label'],v.get('original',''),v['reason']) for v in result['findings'] }
    for raw in items:
        if not isinstance(raw,dict) or raw.get('label') not in SEMANTIC_LABELS:raise ValueError('建议包含停用或未知检查类别。')
        item={key:raw[key] for key in ('field','label','reason','original','replacement','occurrence') if key in raw}
        if not isinstance(item.get('reason'),str) or not item['reason'].strip():raise ValueError('建议必须说明理由。')
        if rules.DISABLED_FINDING_RE.search(item['reason']):raise ValueError('停用检查项不能通过其他类别报告。')
        if item.get('field') not in sections(report['caption']):raise ValueError('语义建议字段不存在。')
        item.setdefault('original','');item.setdefault('replacement',None);item['decision']='pending';item['source']='semantic'
        if item['original'] and item['original'] not in sections(report['caption'])[item['field']]['text']:raise ValueError('节选不是当前稿中的原文。')
        if item['replacement'] is not None:item['operation']=candidate_operation(report['caption'],item)
        key=(item['field'],item['label'],item['original'],item['reason'])
        if key not in existing:result['findings'].append(item);existing.add(key)
    result['semantic_status']=proposal['semantic_status'];bind_findings(result)
    result['status']='NEEDS_CONFIRMATION' if result['findings'] else 'SCREENING_NO_PENDING' if result['semantic_status']=='completed' else 'SEMANTIC_INCOMPLETE'
    return result


def apply(report,ids,confirmation,allow_audio=False):
    if not confirmation.strip():raise ValueError('必须记录用户明确确认的原话，不能代替用户批准。')
    if not ids or len(ids)!=len(set(ids)):raise ValueError('须指定不重复的问题编号。')
    by_id={i['issue_id']:i for i in report['findings']};ops={};history=[]
    for identifier in ids:
        if identifier not in by_id:raise ValueError('问题编号不存在：'+identifier)
        item=by_id[identifier];op=candidate_operation(report['caption'],item)
        if op['audio_confirmation_required'] and not allow_audio:raise ValueError('涉及音频内部标点，须核对素材并明确同意，再加--allow-audio-punctuation。')
        ops.setdefault(item['field'],[]).append(op)
        history.append({'issue_id':identifier,'field':item['field'],'operation':op,'confirmation':confirmation})
    values=sections(report['caption']);caption=report['caption'];field_edits=[]
    for field,items in ops.items():
        ordered=sorted(items,key=lambda o:(o['start'],o['end']))
        if any(a['end']>b['start'] for a,b in zip(ordered,ordered[1:])):raise ValueError('确认项位置重叠，须合并成一个明确改稿后再确认。')
        text=values[field]['text']
        for op in reversed(ordered):text=text[:op['start']]+op['new']+text[op['end']:]
        field_edits.append((values[field]['start'],values[field]['end'],text))
    for a,b,text in sorted(field_edits,reverse=True):caption=caption[:a]+text+caption[b:]
    if list(sections(caption))!=list(values):raise ValueError('合并后字段边界变化，所有确认项均未应用。')
    # Joint numeric/reference/protected checks catch interactions across approved edits.
    for field in ops:
        union=finding(field,'主体引用' if any(by_id[i]['label']=='主体引用' and by_id[i]['field']==field for i in ids) else '疑似病句',
            '合并校验',values[field]['text'],sections(caption)[field]['text'])
        candidate_operation(report['caption'],union)
    result=check(caption,report.get('sample_id'))
    result['parent_sha256']=report['base_sha256'];result['approved_history']=report.get('approved_history',[])+history
    result['next_issue_number']=report['next_issue_number']
    old_ids={(v['field'],v['label'],v.get('original',''),v['reason']):v['issue_id'] for v in report['findings'] if v['issue_id'] not in ids}
    for item in result['findings']:
        item.pop('issue_id',None)
        key=(item['field'],item['label'],item.get('original',''),item['reason'])
        if key in old_ids:item['issue_id']=old_ids[key]
    result['semantic_status']='needs_recheck'
    for item in report['findings']:
        if item['issue_id'] in ids or item.get('source')!='semantic':continue
        carry={k:v for k,v in item.items() if k not in {'operation','excerpt','subject'}}
        if carry.get('replacement') is not None:
            try:carry['operation']=candidate_operation(result['caption'],carry)
            except ValueError:
                carry['replacement']=None;carry['reason']+='；当前稿已变化，原候选待重新核对。'
        result['findings'].append(carry)
    bind_findings(result);result['status']='NEEDS_CONFIRMATION' if result['findings'] else 'LOCAL_RECHECK_COMPLETE'
    return result


def markdown(report):
    lines=['# 单段 Caption 待确认报告','', '题目 ID：'+str(report.get('sample_id') or '未提供（不补造）'),
           '状态：'+report['status'],'语义检查：'+report['semantic_status'],'']
    if not report['findings']:lines+=['本轮未发现需要人工确认的问题。' if report['semantic_status']=='completed' else '本地规则没有待确认问题；语义检查尚未完成。','']
    for item in report['findings']:
        lines += ['## '+item['issue_id']+'｜'+item['label'],'字段：'+item['field'],'理由：'+item['reason'],'']
        if item.get('excerpt'):lines+=['原句：','```text',item['excerpt'],'```']
        if item.get('replacement') is not None:lines+=['建议替换为：','```text',item['replacement'],'```']
        if item.get('subject'):lines+=['主体字段：','```text',item['subject'].strip(),'```']
        if item.get('operation',{}).get('audio_confirmation_required'):lines+=['涉及音频内部标点：须先核对素材并明确确认。']
        lines.append('')
    return '\n'.join(lines)+'\n'


def write_result(report,out_dir):
    folder=Path(out_dir)/('caption-'+uuid4().hex[:12]);folder.mkdir(parents=True,exist_ok=False)
    for name,value in {'caption.txt':report['caption'],'report.json':json.dumps(report,ensure_ascii=False,indent=2)+'\n','待确认报告.md':markdown(report)}.items():
        with (folder/name).open('x',encoding='utf-8',newline='') as stream:stream.write(value)
    print(json.dumps({'status':report['status'],'semantic_status':report['semantic_status'],'pending':len(report['findings']),
        'caption':str((folder/'caption.txt').resolve()),'report':str((folder/'report.json').resolve()),
        'review':str((folder/'待确认报告.md').resolve())},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    c=subs.add_parser('check');c.add_argument('--input',required=True);c.add_argument('--id');c.add_argument('--check-only',action='store_true')
    p=subs.add_parser('propose');p.add_argument('--report',required=True);p.add_argument('--suggestions',required=True)
    a=subs.add_parser('apply');a.add_argument('--report',required=True);a.add_argument('--approve',nargs='+',required=True);a.add_argument('--confirmation',required=True);a.add_argument('--allow-audio-punctuation',action='store_true')
    for item in (c,p,a):item.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    try:
        if args.command=='check':report=check(read_text(args.input),args.id,args.check_only)
        elif args.command=='propose':report=propose(read_report(args.report),json_file(args.suggestions))
        else:report=apply(read_report(args.report),args.approve,args.confirmation,args.allow_audio_punctuation)
        write_result(report,args.out_dir);return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(json.dumps({'status':'NOT_COMPLETED','reason':str(exc),'source_overwritten':False},ensure_ascii=False),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
