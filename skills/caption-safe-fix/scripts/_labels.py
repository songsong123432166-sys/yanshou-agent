"""Recognized metadata separators only; no semantic label inference."""
import re
from _schema import TASK_LABELS

POLICY='label-separator-v2'

LEGACY_POLICY='label-separator-v1'

PIPE_POLICY='fullwidth-pipe-v1'

FIELDS={'任务类型','task_type','subtask_type','video_similarity_tag','similarity_tag',
        'audio_similarity_tag','视听标签'}

SEPARATOR=re.compile(r'\r\n|[\r\n|｜、，,；;]')

VISUAL=r'实体(?:/ip)?|人物|物品|场景|动作|镜头(?:/运镜)?|运镜|剪辑(?:方式)?|剧情|布局|视角|视觉风格|动态|特效'

AUDIO=r'音色|音乐|旋律|音效|环境音|台词(?:内容)?|人声|声音'

RELATION=r'相同|同个|同类不同个|不同'

def _separator_operations(field, text, legacy=False):
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
    target=next((m[0] for m in matches if m[0] in {'|','｜'}),'｜') if legacy else '｜'
    return [{'start':leading+m.start(),'end':leading+m.end(),'old':m[0],'new':target}
            for m in matches if m[0] not in ({'|','｜'} if legacy else {'｜'})]

def separator_operations(field,text):return _separator_operations(field,text)

def pipe_width_operations(field,text):
    if field not in FIELDS or not isinstance(text,str):return []
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return []
    from caption_qa_width import protected_spans, TECH
    protected=protected_spans(text)+[m.span() for m in TECH.finditer(text)]
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    return [{'start':m.start(),'end':m.end(),'old':'|','new':'｜'} for m in re.finditer(r'\|',text)
            if not any(a<=m.start()<b for a,b in protected)]

def normalize_pipe_width(field,text):
    for op in reversed(pipe_width_operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text

def normalize_separators(field,text):
    value=text
    for op in reversed(separator_operations(field,text)):
        value=value[:op['start']]+op['new']+value[op['end']:]
    return normalize_pipe_width(field,value)
