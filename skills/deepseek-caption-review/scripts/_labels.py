"""Recognized metadata separators only; no semantic label inference."""
import re
from _schema import TASK_LABELS

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
        from _metadata import normalize as translate_tags
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

def _pipe_width_operations(field,text,previous=False):
    if field not in FIELDS or not isinstance(text,str):return []
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return []
    from caption_qa_width import protected_spans, TECH
    from _spacing import operations, operations as previous_operations
    protected=protected_spans(text)+[m.span() for m in TECH.finditer(text)]
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    target='｜' if previous else '|'
    edits=[{'start':m.start(),'end':m.end(),'old':m[0],'new':target}
        for m in re.finditer(r'[ \t\u3000]*[|｜][ \t\u3000]*',text)
        if m[0]!=target and not any(m.start()<b and m.end()>a for a,b in protected)]
    return sorted(edits+(previous_operations if previous else operations)(field,text),key=lambda op:(op['start'],op['end']))

def pipe_width_operations(field,text):return _pipe_width_operations(field,text)

def normalize_pipe_width(field,text):
    for op in reversed(pipe_width_operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    from _metadata import case_operations
    for op in reversed(case_operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text

def normalize_separators(field,text):
    value=text
    for op in reversed(separator_operations(field,text)):
        value=value[:op['start']]+op['new']+value[op['end']:]
    return normalize_pipe_width(field,value)
