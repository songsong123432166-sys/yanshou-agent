"""Context-bound width normalization; no NFKC or blanket text conversion."""
import re

TAG = re.compile(r'[<＜]\s*(?:主体|场景|图片|视频|音频|画面元素)\s*\d+\s*[>＞]')
BRACES = re.compile(r'[{｛][^{}｛｝]*(?:[}｝]|$)')
SCREEN = re.compile(r'【[^【】]*(?:】|$)')
ANGLE = re.compile(r'[<＜][^<>＜＞\n]*[>＞]')
MUSIC = re.compile(r'^\s*(?:背景(?:音乐|配乐)|背景中.*?(?:播放|音乐)|配乐|BGM\b|音乐(?:是|为|继续|渐|淡))',re.I)
SOUND = re.compile(r'音效|环境音|声音|声响|传来.*声|发出.*声|[鸣响叫笑哭喊唱]声|摩擦声|燃烧声')
TECH = re.compile(r'(?:https?://|www\.)[^\s<＜【｛{，。；！？、（）]+|[\w.+-]+@[\w.-]+\.[A-Za-z]+|'
                  r'(?<!\w)\d+(?:(?:[.:/-])\d+)+(?!\w)|'
                  r'(?<!\w)(?:[A-Za-z]:[\\/]|/|\./|\.\./)[\w./\\:+?=&%-]+|'
                  r'\b[A-Za-z_][\w.-]*\.[A-Za-z]{1,8}\b|`[^`]*`')
CHINESE = re.compile(r'[\u3400-\u9fff]')


def unsafe_regions(name,text):
    """Incomplete wrappers need a human; do not invent their missing half."""
    regions=[]
    literals=[(m.start(),m.end()) for m in SCREEN.finditer(text)]
    for match in BRACES.finditer(text):
        if match[0].startswith('｛') and match[0][-1] not in '}｝':
            regions.append((match.start(),match.end()))
    complete=[(m.start(),m.end()) for m in TAG.finditer(text)]
    pattern=r'＜\s*(?:主体|场景|图片|视频|音频|画面元素)\s*\d+\s*|(?:主体|场景|图片|视频|音频|画面元素)\s*\d+\s*＞'
    for match in re.finditer(pattern,text):
        if not any(a<=match.start() and match.end()<=b for a,b in complete):
            regions.append((match.start(),match.end()))
    if name=='自然指令':
        stack=[]
        for pos,char in enumerate(text):
            if char in '(（':stack.append(pos)
            elif char in ')）' and stack:stack.pop()
        regions.extend((pos,len(text)) for pos in stack if MUSIC.search(text[pos+1:]))
        for match in re.finditer(r'＜[^<>＜＞\n]*(?=\n|$)',text):
            if SOUND.search(match[0]):regions.append((match.start(),match.end()))
    return [(a,b) for a,b in regions if not any(x<=a and b<=y for x,y in literals)]


def operations(name, text, *, exempt=False):
    """Return independently verifiable, same-length edits in original offsets."""
    if exempt:
        return []
    protected, wrappers = [], []
    # Screen words and speech bodies are opaque, even with full-width braces.
    for match in SCREEN.finditer(text):
        protected.append((match.start(),match.end()))
    for match in BRACES.finditer(text):
        protected.append((match.start(),match.end()))
        if match[0][-1] in '}｝':
            wrappers.extend([(match.start(),'{','人声包裹符'),(match.end()-1,'}','人声包裹符')])
    stack=[]
    for pos,char in enumerate(text):
        if char in '（(':
            stack.append(pos)
        elif char in '）)' and stack:
            start=stack.pop()
            if name=='自然指令' and MUSIC.search(text[start+1:pos]):
                protected.append((start,pos+1))
                wrappers.extend([(start,'（','背景音乐外层括号'),(pos,'）','背景音乐外层括号')])
    # Do not edit punctuation inside an unclosed audio description.
    for start in stack:
        if name=='自然指令' and MUSIC.search(text[start+1:]):
            protected.append((start,len(text)))
    for match in ANGLE.finditer(text):
        if TAG.fullmatch(match[0]):
            wrappers.extend([(match.start(),'<','编号标签尖括号'),(match.end()-1,'>','编号标签尖括号')])
        elif name=='自然指令' and SOUND.search(match[0]):
            protected.append((match.start(),match.end()))
            wrappers.extend([(match.start(),'<','音效包裹符'),(match.end()-1,'>','音效包裹符')])
    protected.extend(unsafe_regions(name,text))
    # Protect literal quoted strings and technical tokens from prose rules.
    technical=[(m.start(),m.end()) for m in TECH.finditer(text)]
    quoted=[(m.start(),m.end()) for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」',text)]
    result=[]
    for pos,new,rule in wrappers:
        # Only the outer wrapper of a protected fragment may be normalized.
        # Wrappers nested inside speech, BGM or screen text stay untouched.
        if any(a < pos < b-1 for a,b in protected) or any(a<=pos<b for a,b in technical+quoted):
            continue
        if text[pos]!=new:
            result.append({'start':pos,'end':pos+1,'old':text[pos],'new':new,'width_rule':rule})
    mapping={',':'，',';':'；',':':'：','!':'！','?':'？','.':'。'}
    offset=0
    for line in text.splitlines(keepends=True):
        for local,char in enumerate(line):
            if char not in mapping:
                continue
            pos=offset+local
            if any(a<=pos<b for a,b in protected+technical+quoted):
                continue
            before=line[:local].rstrip();after=line[local+1:].lstrip()
            left=before[-1:] or '';right=after[:1] or ''
            # Chinese prose adjacency only; English punctuation is preserved.
            chinese_adjacent=bool(CHINESE.search(left+right))
            closed_chinese=left in '>＞}｝】）)”' and bool(CHINESE.search(before))
            if not chinese_adjacent and not closed_chinese:
                continue
            if char=='.':
                if right=='.' or left=='.' or (left.isdigit() and right.isdigit()):
                    continue
                if after and not CHINESE.match(after):
                    continue
            if char==':' and left.isdigit() and right.isdigit():
                continue
            result.append({'start':pos,'end':pos+1,'old':char,'new':mapping[char],'width_rule':'中文叙述标点'})
        offset+=len(line)
    unique={(o['start'],o['end'],o['old'],o['new']):o for o in result}
    return [unique[key] for key in sorted(unique)]


def normalize(name,text,*,exempt=False):
    edits=operations(name,text,exempt=exempt)
    for op in reversed(edits):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text


def valid_operation(name,text,op,*,exempt=False):
    return any(all(candidate.get(key)==op.get(key) for key in ('start','end','old','new','width_rule'))
               for candidate in operations(name,text,exempt=exempt))
