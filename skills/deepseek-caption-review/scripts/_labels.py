"""Recognized metadata separators only; no semantic label inference."""
import re
from _schema import TASK_LABELS

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
            return re.sub(r'\s*-\s*','-',value) in TASK_LABELS
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
