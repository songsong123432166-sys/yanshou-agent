"""Shared field schema snapshot. No model client or network operations."""
import re
from _metadata import TITLES, ALIASES as TAG_ALIASES, normalize as translate_tags

VERSION=11

HEADER_PREFIX='## '

FIELDS=('任务类型','提示','similarity_label','video_similarity_tag','audio_similarity_tag',
        '输入描述','基调','场景','主体','画面元素','总结','自然指令')

ALIASES={name:name for name in FIELDS}

ALIASES.update(TAG_ALIASES)

ALIASES.update(task_type='_task',subtask_type='_subtask',brief_captions='输入描述',subject='主体',
               natural_instruction='自然指令',similarity_tag='video_similarity_tag',
               tone='基调',scene='场景',visual_element='画面元素',summary='总结',
               **{'自然指令（细节）':'自然指令','自然指令(细节)':'自然指令','自然描述':'自然指令'})

TASK_LABELS=('人物参考','物品参考','造型参考','场景参考','动作参考','布局参考','运镜参考',
             '剧情参考','剪辑参考','视角参考','视觉风格参考','创意参考','动态参考',
             '俯视图/全景图参考','复合音频参考','旋律参考','音色参考','续写','向前续写',
             'prompt图参考','向前延长','向后延长','向前延长后切镜','切镜后延长',
             '指令编辑-严格编辑-增','指令编辑-严格编辑-删','指令编辑-严格编辑-改',
             '参考-严格编辑-增','参考-严格编辑-改','创意编辑','参考-创意编辑',
             '首帧控制','尾帧控制','首尾帧控制','多帧控制','九宫格','少宫格','白模')

PATTERN=re.compile(r'^\s*(?P<mark>#{1,6}\s*)?(?P<name>'+ '|'.join(re.escape(n) for n in sorted(ALIASES,key=len,reverse=True))+r')(?:(?:\s*[:：]\s*)(?P<inline>.*)|[ \t]*[。.]?)$')

PROTECTED=re.compile(r'[{｛][^{}｛｝]*(?:[}｝]|$)|【[^【】]*(?:】|$)',re.S)

def _units(text):
    """Split prose only outside quoted speech, displayed words and wrappers."""
    stack=[];start=0
    pairs={'{':'}','｛':'｝','【':'】','<':'>','＜':'＞','（':'）','(':')','“':'”','「':'」'}
    for n,char in enumerate(text):
        if stack and stack[-1] in ('}','｝','】'):
            if char==stack[-1]:stack.pop()
        elif char in pairs:stack.append(pairs[char])
        elif stack and char==stack[-1]:stack.pop()
        if not stack and char in '。！？\n':
            if text[start:n+1].strip():yield text[start:n+1].strip()
            start=n+1
    if text[start:].strip():yield text[start:].strip()

def parse(text):
    if not isinstance(text,str) or not text.strip():
        return {'headers':[],'blocks':[],'unknown':[]}
    spans=[m.span() for m in PROTECTED.finditer(text)]
    headers=[];blocks=[];unknown=[];current=None;raw='';offset=0;alias=None
    def flush():
        nonlocal raw
        for value in _units(raw):blocks.append({'id':len(blocks),'field':current,'alias':alias,'text':value})
        raw=''
    for line in text.splitlines(keepends=True):
        start=offset;value=line.rstrip('\r\n');position=offset+len(value)-len(value.lstrip());offset+=len(line)
        match=None if any(a<=position<b for a,b in spans) else PATTERN.fullmatch(value)
        if match:
            flush();alias=match['name'];current=ALIASES[alias]
            headers.append({'field':current,'alias':alias,'canonical':value.rstrip(' \t')==HEADER_PREFIX+TITLES.get(current,current),
                            'start':start,'end':offset,'line':line,'inline':match['inline']})
            if match['inline']:raw=match['inline']+'\n'
        else:
            if value.lstrip().startswith('#') and not any(a<=position<b for a,b in spans):unknown.append(value)
            raw+=line
    flush()
    return {'headers':headers,'blocks':blocks,'unknown':unknown}

def is_legacy_reference(text):
    """Match the old task_type value exactly, never prose or a newer label."""
    return any(block['field']=='_task' and block['text'].strip()=='参考' for block in parse(text)['blocks'])

def metadata_value(name,text):
    from _labels import normalize_separators
    return normalize_separators(name,translate_tags(name,text))

def render(fields):
    text='\n\n'.join(HEADER_PREFIX+TITLES.get(name,name)+'\n'+metadata_value(name,fields.get(name,'')).strip() for name in FIELDS)+'\n'
    return space_fields(text)

def space_fields(text,parsed=None):
    """Keep one blank line after each field, without reflowing its body."""
    headers=(parsed or parse(text))['headers']
    if not headers:return text
    newline='\r\n' if '\r\n' in text else '\n'
    chunks=[text[:headers[0]['start']]]
    for index,header in enumerate(headers):
        end=headers[index+1]['start'] if index+1<len(headers) else len(text)
        lines=text[header['start']:end].splitlines(keepends=True)
        while lines and not lines[-1].strip():lines.pop()
        chunk=''.join(lines)
        chunks.append(chunk+(newline if chunk.endswith(('\n','\r')) else newline*2))
    return ''.join(chunks)

def _body(field,text):
    if field not in {'输入描述','主体','场景'}:return text
    # Only old definition prefixes change; literal bodies remain byte-identical.
    return re.sub(r'(?m)^[ \t]*[-*]?[ \t]*(图片|视频|音频|主体|场景)(\d+)[ \t]*[:：][ \t]*',
                  lambda m:'<'+m[1]+m[2]+'>：',text)

def local_caption(text,parsed,task):
    """Edit header lines directly; never reflow already assigned field bodies."""
    newline='\r\n' if '\r\n' in text else '\n'
    sections={};headers=parsed['headers'];prefix=text[:headers[0]['start']]
    notes=[]
    for index,header in enumerate(headers):
        field=header['field']
        if field in {'_task','_subtask'}:continue
        end=headers[index+1]['start'] if index+1<len(headers) else len(text)
        body=text[header['end']:end]
        if header['inline'] is not None:body=header['inline']+newline+body
        if header['alias'] in {'brief_captions','subject'}:body=_body(field,body)
        title=header['line'] if header['canonical'] else HEADER_PREFIX+TITLES.get(field,field)+(newline if header['line'].endswith(('\r','\n')) or body else '')
        sections[field]=title+metadata_value(field,body)
    if '任务类型' not in sections:
        sections['任务类型']=HEADER_PREFIX+'任务类型'+newline+task+newline+newline
    chunks=[]
    for field in FIELDS:
        if field not in sections:
            sections[field]=HEADER_PREFIX+TITLES.get(field,field)+newline+newline
        chunk=sections[field]
        if chunks and not chunks[-1].endswith(('\n','\r')):chunks[-1]+=newline
        chunks.append(chunk)
    caption=space_fields(prefix+''.join(chunks))
    again=parse(caption)
    if [h['field'] for h in again['headers']]!=list(FIELDS) or again['unknown']:
        raise ValueError('整理结果字段边界不规范，未执行转换。')
    empty=[name for name in FIELDS if not any(b['field']==name for b in again['blocks'])]
    if empty:notes.append('以下字段没有正文，保持留空：'+'、'.join(empty))
    return caption,notes

def task_value(parsed,mapping=None):
    values={key:[] for key in ('任务类型','_task','_subtask')}
    for b in parsed['blocks']:
        if b['field'] in values:values[b['field']].append(b['text'].strip())
    old=[v for key in ('_task','_subtask') for v in values[key]]
    if values['任务类型']:
        distinct=list(dict.fromkeys(values['任务类型']))
        if len(distinct)>1:raise ValueError('任务类型存在不同版本，请确认保留哪一份。')
        if old:
            key=' / '.join(old);mapped=(mapping or {}).get(key)
            if mapped is None:raise ValueError('新旧任务类型同时存在，旧值尚无明确映射：'+key+'；保留原行待确认。')
            labels=mapped if isinstance(mapped,list) else re.split('[|｜]',mapped)
            if set(re.split('[|｜]',distinct[0]))!=set(labels):raise ValueError('新旧任务类型映射结果不一致，保留原行待确认。')
        return distinct[0]
    if not old:return ''
    mapping=mapping or {}
    key=' / '.join(old)
    target=mapping.get(key)
    if target is None and len(old)==1:target=mapping.get(old[0])
    if target is None:
        raise ValueError('旧任务类型尚无新版映射：'+key+'；请在设置中填写任务标签映射。')
    labels=target if isinstance(target,list) else re.split('[|｜]',target)
    if not labels or any(not isinstance(s,str) or not s.strip() for s in labels):raise ValueError('任务标签映射须为非空标签或标签列表。')
    return '|'.join(s.strip() for s in labels)

def is_generation_request(text):
    """Recognize explicit prompt prose, without treating old subtype tags as prose."""
    value=text.strip()
    return (bool(value) and value not in TASK_LABELS and
            not re.match(r'^(?:参考|编辑|延长|续写)[ \t]*[-/]',value) and
            bool(re.match(r'^(?:以|基于|根据|参考|将|把|请|补全|续写|向前续写|向前延长|延长|编辑|严格编辑|生成|制作)',value)) and
            bool(re.search(r'生成|制作|补全|续写|延长|编辑',value)))
