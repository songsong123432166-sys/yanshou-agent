"""Deterministic separator fixes for complete, recognizable metadata labels."""
import re
from captionqa.caption_field_schema import TASK_LABELS

POLICY='label-separator-v3'
PREVIOUS_POLICY='label-separator-v2'
LEGACY_POLICY='label-separator-v1'
PIPE_POLICY='label-delimiter-format-v3'
FIELDS={'任务类型','task_type','subtask_type','video_similarity_tag','similarity_tag',
        'audio_similarity_tag','视听标签'}
SEPARATOR=re.compile(r'\r\n|[\r\n|｜、，,；;]')
VISUAL=r'实体(?:/ip)?|人物|物品|场景|动作|镜头(?:/运镜)?|运镜|剪辑(?:方式)?|剧情|布局|视角|视觉风格|动态|特效'
AUDIO=r'音色|音乐|旋律|音效|环境音|台词(?:内容)?|人声|声音'
RELATION=r'相同|同个|同类不同个|不同'


def _separator_operations(field, text, legacy=False, previous=False):
    """Only replace separators; keep label spelling, order and all other text."""
    if field not in FIELDS or not isinstance(text,str) or not text.strip():return []
    leading=len(text)-len(text.lstrip());body=text.strip()
    # The comma in an English relation is part of its value. Mask it with
    # one character so original operation offsets remain valid.
    view=re.sub(r'Same[ \t]+Category,[ \t]*Different[ \t]+Instance',lambda m:m[0].replace(',', '\ufff0'),body,flags=re.I)
    matches=list(SEPARATOR.finditer(view))
    if not matches:return []
    labels=SEPARATOR.split(view)
    if any(not label.strip() for label in labels):return []
    def valid(label):
        from captionqa.caption_qa_metadata import normalize as translate_tags
        value=translate_tags(field,label.replace('\ufff0',',').strip())
        if field in {'任务类型','task_type','subtask_type'}:
            return re.sub(r'\s*-\s*','-',value) in TASK_LABELS
        dimensions=AUDIO if field=='audio_similarity_tag' else VISUAL if field in {'video_similarity_tag','similarity_tag'} else VISUAL+'|'+AUDIO
        return bool(re.fullmatch(r'(?:'+dimensions+r')\s*-\s*(?:'+RELATION+r')',value,re.I))
    if not all(valid(label) for label in labels):return []
    target=next((m[0] for m in matches if m[0] in {'|','｜'}),'｜') if legacy else '｜' if previous else '|'
    return [{'start':leading+m.start(),'end':leading+m.end(),'old':m[0],'new':target}
            for m in matches if m[0] not in ({'|','｜'} if legacy else {target})]


def separator_operations(field,text):return _separator_operations(field,text)


def legacy_separator_operations(field,text):return _separator_operations(field,text,legacy=True)


def previous_separator_operations(field,text):return _separator_operations(field,text,previous=True)


def legacy_pipe_width_operations(field,text):
    if field not in FIELDS or not isinstance(text,str):return []
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return []
    from captionqa.caption_qa_width import protected_spans, TECH
    protected=protected_spans(text)+[m.span() for m in TECH.finditer(text)]
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    return [{'start':m.start(),'end':m.end(),'old':'|','new':'｜'} for m in re.finditer(r'\|',text)
            if not any(a<=m.start()<b for a,b in protected)]


def _pipe_width_operations(field,text,previous=False):
    if field not in FIELDS or not isinstance(text,str):return []
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return []
    from captionqa.caption_qa_width import protected_spans, TECH
    from captionqa.caption_qa_spacing import operations, previous_operations
    protected=protected_spans(text)+[m.span() for m in TECH.finditer(text)]
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    target='｜' if previous else '|'
    edits=[{'start':m.start(),'end':m.end(),'old':m[0],'new':target}
        for m in re.finditer(r'[ \t\u3000]*[|｜][ \t\u3000]*',text)
        if m[0]!=target and not any(m.start()<b and m.end()>a for a,b in protected)]
    return sorted(edits+(previous_operations if previous else operations)(field,text),key=lambda op:(op['start'],op['end']))


def pipe_width_operations(field,text):return _pipe_width_operations(field,text)


def previous_pipe_width_operations(field,text):return _pipe_width_operations(field,text,previous=True)


def normalize_pipe_width(field,text):
    for op in reversed(pipe_width_operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    from captionqa.caption_qa_metadata import case_operations
    for op in reversed(case_operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text


def normalize_separators(field,text):
    value=text
    for op in reversed(separator_operations(field,text)):
        value=value[:op['start']]+op['new']+value[op['end']:]
    return normalize_pipe_width(field,value)


def enrich_separator_fixes(document):
    """Upgrade pending saved findings without undoing any human decisions."""
    from captionqa.caption_qa_review_model import AUTO_WORKFLOWS
    fields={field['field_id']:field for field in document['fields']};changed=[]
    automatic=document['workflow_version'] in AUTO_WORKFLOWS
    for issue in document['issues']:
        if issue.get('kind')!='label_separator' or issue.get('origin')!='static':continue
        if issue['decision'] not in {'pending','defer'} or issue['field_id'] in document.get('field_edits',{}):continue
        field=fields[issue['field_id']]
        operations=separator_operations(field['name'],field['original'])
        if not operations:continue
        selected=[op for other in document['issues'] if other is not issue and other['field_id']==issue['field_id']
                  and other['decision'] in {'auto','approve'} for op in other['operations']]
        if any(a['start']==b['start'] or a['start']<b['end'] and b['start']<a['end'] for a in operations for b in selected):continue
        if issue.get('separator_policy')==POLICY and issue.get('operations')==operations:continue
        issue.update(operations=operations,can_auto_fix=True,separator_policy=POLICY,
                     requires_confirmation=False,reason='标签完整，仅将分隔符规范为半角竖线 |；标签内容与顺序保留。')
        if automatic and issue.get('sample_id'):
            issue['decision']='auto'
        changed.append(issue['issue_id'])
    return changed


def enrich_pipe_width(document,records):
    """Add width-only fixes to old PASS rows without replacing saved drafts."""
    import hashlib
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_review_model import AUTO_WORKFLOWS
    changed=[]
    for index,record in records.items():
        caption=record.get(document['caption_column'])
        if not isinstance(caption,str) or not qa.is_human_caption(caption):continue
        try:sections=qa.parse_caption_sections(caption)
        except ValueError:continue
        for name,original in sections.items():
            ops=pipe_width_operations(name,original)
            if not ops:continue
            field=next((f for f in document['fields'] if f['row_index']==index and f['name']==name),None)
            if field and field['original']!=original:continue
            if field is None:
                raw=record.get(document['id_column']);sample_id=str(raw) if type(raw) in (str,int) else ''
                field={'field_id':hashlib.sha256(f"{document['source_sha256']}:{index}:{name}".encode()).hexdigest()[:24],
                    'row_index':index,'sample_id':sample_id,'name':name,'original':original,'issue_ids':[],
                    'source_sheet':record.get('source_sheet'),'source_row':record.get('source_row')}
                document['fields'].append(field)
            if any(i['field_id']==field['field_id'] and i['kind'] in {'pipe_width','label_separator'} and i['operations'] for i in document['issues']):continue
            key=hashlib.sha256(f"{field['field_id']}:{PIPE_POLICY}".encode()).hexdigest()[:24]
            decision='auto' if document['workflow_version'] in AUTO_WORKFLOWS and field['sample_id'] else 'pending'
            issue={'issue_id':key,'field_id':field['field_id'],'sample_id':field['sample_id'],'row_index':index,
                'field':name,'kind':'pipe_width','label':'标签分隔符格式','snippet':original.strip(),
                'excerpts':[original.strip()],'subject':'','reason':'标签用半角竖线 | 连接，竖线及标签内 - 两侧不留空格。',
                'decision':decision,'operations':ops,'locations':[[o['start'],o['end']] for o in ops],
                'can_auto_fix':True,'origin':'static','source_issue_index':None}
            document['issues'].append(issue);field['issue_ids'].append(key);changed.append(key)
    return changed
