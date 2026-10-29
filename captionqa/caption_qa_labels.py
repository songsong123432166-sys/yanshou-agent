"""Deterministic separator fixes for complete, recognizable metadata labels."""
import re
from captionqa.caption_field_schema import TASK_LABELS

POLICY='label-separator-v1'
FIELDS={'任务类型','task_type','subtask_type','video_similarity_tag','similarity_tag',
        'audio_similarity_tag','视听标签'}
SEPARATOR=re.compile(r'\r\n|[\r\n|｜、，,；;]')
VISUAL=r'实体(?:/ip)?|人物|物品|场景|动作|镜头(?:/运镜)?|运镜|剪辑(?:方式)?|剧情|布局|视角|视觉风格|动态|特效'
AUDIO=r'音色|音乐|旋律|音效|环境音|台词(?:内容)?|人声|声音'
RELATION=r'相同|同个|同类不同个|不同'


def separator_operations(field, text):
    """Only replace separators; keep label spelling, order and all other text."""
    if field not in FIELDS or not isinstance(text,str) or not text.strip():return []
    leading=len(text)-len(text.lstrip());body=text.strip()
    matches=list(SEPARATOR.finditer(body))
    if not matches:return []
    labels=SEPARATOR.split(body)
    if any(not label.strip() for label in labels):return []
    def valid(label):
        value=label.strip()
        if field in {'任务类型','task_type','subtask_type'}:
            return value in TASK_LABELS
        dimensions=AUDIO if field=='audio_similarity_tag' else VISUAL if field in {'video_similarity_tag','similarity_tag'} else VISUAL+'|'+AUDIO
        return bool(re.fullmatch(r'(?:'+dimensions+r')\s*-\s*(?:'+RELATION+r')',value,re.I))
    if not all(valid(label) for label in labels):return []
    target=next((m[0] for m in matches if m[0] in {'|','｜'}),'｜')
    return [{'start':leading+m.start(),'end':leading+m.end(),'old':m[0],'new':target}
            for m in matches if m[0] not in {'|','｜'}]


def normalize_separators(field,text):
    value=text
    for op in reversed(separator_operations(field,text)):
        value=value[:op['start']]+op['new']+value[op['end']:]
    return value


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
                     requires_confirmation=False,reason='标签完整，仅将分隔符规范为竖线；标签内容与顺序保留。')
        if automatic and issue.get('sample_id'):
            issue['decision']='auto'
        changed.append(issue['issue_id'])
    return changed
