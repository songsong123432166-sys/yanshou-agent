"""Single ASCII spaces around time/label hyphen separators, never literals."""
import re

POLICY='hyphen-spacing-v1'
META={'任务类型','task_type','subtask_type','video_similarity_tag','similarity_tag','audio_similarity_tag','视听标签'}
PROSE={'提示','输入描述','基调','场景','主体','画面元素','总结','自然指令'}
TIME=re.compile(r'(?<![A-Za-z0-9.\-−])(?:\d{1,2}:\d{2}(?:\.\d+)?[ \t\u3000]*-[ \t\u3000]*\d{1,2}:\d{2}(?:\.\d+)?|'
    r'\d+(?:\.\d+)?[ \t\u3000]*(?:秒[ \t\u3000]*)?-[ \t\u3000]*\d+(?:\.\d+)?[ \t\u3000]*(?:秒|s\b))',re.I)
LABEL=re.compile(r'[\w/]+(?:[ \t\u3000]*-[ \t\u3000]*[\w/]+)+')
HYPHEN=re.compile(r'[ \t\u3000]*-[ \t\u3000]*')


def operations(field,text):
    if field not in META|PROSE:return []
    from captionqa.caption_qa_width import protected_spans, TECH
    ranges=[]
    if field in META:
        for match in re.finditer(r'[^|｜、，,；;\r\n]+',text):
            if LABEL.fullmatch(match[0].strip()) and re.search(r'[\u3400-\u9fff]',match[0]):ranges.append(match)
    else:
        ranges=list(TIME.finditer(text))
    blocked=protected_spans(text)+[m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    blocked += [m.span() for m in TECH.finditer(text) if not any(r.start()<=m.start() and m.end()<=r.end() for r in ranges)]
    result=[]
    for match in ranges:
        for hyphen in HYPHEN.finditer(match[0]):
            if field in META:
                left=re.search(r'([\w/]+)$',match[0][:hyphen.start()])
                right=re.match(r'([\w/]+)',match[0][hyphen.end():])
                if left and right and all(re.fullmatch(r'[A-Za-z0-9_/]+',s) for s in (left[1],right[1])):continue
            a,b=match.start()+hyphen.start(),match.start()+hyphen.end()
            if hyphen[0]==' - ' or any(a<q and b>p for p,q in blocked):continue
            result.append({'start':a,'end':b,'old':hyphen[0],'new':' - ','spacing_policy':POLICY})
    return result


def normalize(field,text):
    for op in reversed(operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text

