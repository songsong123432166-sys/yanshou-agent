"""Compact metadata-label hyphens only; retain historical operation validation."""
import re

POLICY='metadata-hyphen-spacing-v3'
META={'任务类型','task_type','subtask_type','video_similarity_tag','similarity_tag','audio_similarity_tag','视听标签'}
LABEL=re.compile(r'[\w/]+(?:[ \t\u3000]*-[ \t\u3000]*[\w/]+)+')
HYPHEN=re.compile(r'[ \t\u3000]*-[ \t\u3000]*')


def _operations(field,text,previous=False):
    if field not in META:return []
    from captionqa.caption_qa_width import protected_spans, TECH
    ranges=[]
    for match in re.finditer(r'[^|｜、，,；;\r\n]+',text):
        if LABEL.fullmatch(match[0].strip()) and re.search(r'[\u3400-\u9fff]',match[0]):ranges.append(match)
    blocked=protected_spans(text)+[m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    blocked += [m.span() for m in TECH.finditer(text) if not any(r.start()<=m.start() and m.end()<=r.end() for r in ranges)]
    result=[]
    for match in ranges:
        for hyphen in HYPHEN.finditer(match[0]):
            left=re.search(r'([\w/]+)$',match[0][:hyphen.start()])
            right=re.match(r'([\w/]+)',match[0][hyphen.end():])
            if left and right and all(re.fullmatch(r'[A-Za-z0-9_/]+',s) for s in (left[1],right[1])):continue
            a,b=match.start()+hyphen.start(),match.start()+hyphen.end()
            target=' - ' if previous else '-'
            if hyphen[0]==target or any(a<q and b>p for p,q in blocked):continue
            result.append({'start':a,'end':b,'old':hyphen[0],'new':target,
                           'spacing_policy':'metadata-hyphen-spacing-v2' if previous else POLICY})
    return result


def operations(field,text):return _operations(field,text)


def previous_operations(field,text):return _operations(field,text,previous=True)


def normalize(field,text):
    for op in reversed(operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text

