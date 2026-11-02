"""Portable offline rule support; no application or API imports."""
import re


POLICY='time-range-paragraph-v1'

RANGE=re.compile(r'(?<![\w.<>])(?:第[ \t]*)?\d+(?:\.\d+)?[ \t]*(?:秒[ \t]*)?'
    r'[-~～—–至到][ \t]*\d+(?:\.\d+)?[ \t]*(?:秒|s\b)(?=[ \t]*(?:[，,:：]|$))',re.I)

def operations(field,text):
    if field!='自然指令':return []
    from _audio import audio_spans
    from _width_literals import SCREEN, TECH
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
