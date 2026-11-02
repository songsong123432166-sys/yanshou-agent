"""Deterministic paragraph layout for explicit natural-instruction time ranges.

Only whitespace changes. Point timestamps (第2秒) and protected audio/screen
content are opaque; no time ordering or duration judgments are performed.
"""
import hashlib
import re

POLICY='time-range-paragraph-v1'
RANGE=re.compile(r'(?<![\w.<>])(?:第[ \t]*)?\d+(?:\.\d+)?[ \t]*(?:秒[ \t]*)?'
    r'[-~～—–至到][ \t]*\d+(?:\.\d+)?[ \t]*(?:秒|s\b)(?=[ \t]*(?:[，,:：]|$))',re.I)


def operations(field,text):
    if field!='自然指令':return []
    from captionqa.caption_qa_review_audio import audio_spans
    from captionqa.caption_qa_width import SCREEN, TECH
    matches=list(RANGE.finditer(text))
    technical=[m.span() for m in TECH.finditer(text)
               if not any(r.start()<=m.start() and m.end()<=r.end() for r in matches)]
    protected=audio_spans(text)+[m.span() for m in SCREEN.finditer(text)]+technical
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    newline='\r\n' if '\r\n' in text else '\n'
    result=[]
    for match in matches:
        if any(a<=match.start()<b for a,b in protected):continue
        gap=re.search(r'[ \t\r\n]*$',text[:match.start()])
        start=gap.start()
        if not text[:start].strip() or gap[0].count('\n')>=2:continue
        result.append({'start':start,'end':match.start(),'old':gap[0],'new':newline*2,
                       'time_range_rule':POLICY,'heading':match[0]})
    return result


def normalize(field,text):
    for op in reversed(operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text


def enrich_time_layout(document,records):
    """Add the new local rule to old reviews; preserve explicit human choices."""
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_review_model import AUTO_WORKFLOWS
    automatic=document['workflow_version'] in AUTO_WORKFLOWS
    changed=[]
    for index,record in records.items():
        caption=record.get(document['caption_column'])
        if not isinstance(caption,str) or not qa.is_human_caption(caption):continue
        try:sections=qa.parse_caption_sections(caption)
        except ValueError:continue
        original=sections.get('自然指令','')
        ops=operations('自然指令',original)
        if not ops:continue
        field=next((f for f in document['fields'] if f['row_index']==index and f['name']=='自然指令'),None)
        if field and (field['field_id'] in document.get('field_edits',{}) or field['original']!=original):continue
        if field is None:
            raw_id=record.get(document['id_column'])
            sample_id=str(raw_id) if type(raw_id) in (str,int) else ''
            field={'field_id':hashlib.sha256(f"{document['source_sha256']}:{index}:自然指令".encode()).hexdigest()[:24],
                'row_index':index,'sample_id':sample_id,'name':'自然指令','original':original,'issue_ids':[],
                'source_sheet':record.get('source_sheet'),'source_row':record.get('source_row')}
            document['fields'].append(field)
        for op in ops:
            existing=[i for i in document['issues'] if i['field_id']==field['field_id'] and i['kind']=='time_range_break'
                      and any(a==op['end'] for a,b in i.get('locations',[]))]
            if existing:continue
            key=hashlib.sha256(f"{field['field_id']}:{POLICY}:{op['end']}".encode()).hexdigest()[:24]
            decision='auto' if automatic and field['sample_id'] else 'pending'
            issue={'issue_id':key,'field_id':field['field_id'],'sample_id':field['sample_id'],
                'row_index':index,'field':'自然指令','kind':'time_range_break','label':'时间段分段',
                'snippet':op['heading'],'excerpts':[original[max(0,op['start']-35):op['end']+70]],'subject':'',
                'reason':'新时间段另起一段，段间空一行；段内第几秒不强制换段。',
                'decision':decision,'operations':[op],'locations':[[op['end'],op['end']+len(op['heading'])]],
                'can_auto_fix':True,'origin':'static','source_issue_index':None}
            document['issues'].append(issue);field['issue_ids'].append(key);changed.append(key)
    return changed
