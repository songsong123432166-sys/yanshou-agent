"""Chinese metadata names and conservative, reversible tag translations."""
import re

TITLES = {'similarity_label': '参考关系标签', 'video_similarity_tag': '视觉标签',
          'audio_similarity_tag': '听觉标签'}
ALIASES = {value: key for key, value in TITLES.items()}
ALIASES.update({'Similarity Label': 'similarity_label',
                'Video Similarity Tag': 'video_similarity_tag',
                'Audio Similarity Tag': 'audio_similarity_tag'})
POLICY = 'chinese-tags-v2'
DIMENSIONS = {
    'entity/ip': '实体/IP', 'entity': '实体', 'character': '人物', 'object': '物品',
    'scene': '场景', 'action': '动作', 'camera/motion': '镜头/运镜',
    'camera movement': '镜头/运镜', 'camera': '镜头/运镜', 'motion': '动态',
    'editing': '剪辑', 'plot': '剧情', 'layout': '布局', 'viewpoint': '视角',
    'visual style': '视觉风格', 'vfx': '特效', 'visual effects': '特效',
    'timbre': '音色', 'music': '音乐', 'melody': '旋律', 'sound effect': '音效',
    'sound effects': '音效', 'ambient sound': '环境音', 'dialogue': '台词内容',
    'voice': '人声', 'sound': '声音',
}
RELATIONS = {'identical': '相同', 'same instance': '同个',
             'same category, different instance': '同类不同个', 'different': '不同'}
_TOKENS = {**DIMENSIONS, **RELATIONS}
_WORDS = '|'.join(re.escape(v).replace(r'\ ', r'[ \t]+') for v in sorted(_TOKENS, key=len, reverse=True))
TOKEN_RE = re.compile(r'(?<![\w/])(?:' + _WORDS + r')(?![\w/])', re.I)


def operations(field, text):
    """Only translate complete tag atoms, never English prose or quoted text."""
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return []
    if field in {'similarity_label','video_similarity_tag','audio_similarity_tag','similarity_tag','视听标签'} and re.fullmatch(r'\s*(yes|no)\s*',text,re.I):
        match=re.fullmatch(r'\s*(yes|no)\s*',text,re.I)
        return [{'start':match.start(1),'end':match.end(1),'old':match[1],
                 'new':'是' if match[1].lower()=='yes' else '否'}] if match else []
    if field not in {'video_similarity_tag', 'similarity_tag', 'audio_similarity_tag', '视听标签'}:
        return []
    edits = []
    # A comma inside the English relation belongs to that atom, not the list.
    for match in re.finditer(r'[^|｜;；\r\n]+', text):
        piece = match[0]
        tokens = list(TOKEN_RE.finditer(piece))
        remainder = TOKEN_RE.sub('', piece)
        remainder = re.sub(r'实体(?:/IP)?|人物|物品|场景|动作|镜头/运镜|特效|音色|音乐|旋律|音效|环境音|人声|同类不同个|相同|同个|不同', '', remainder, flags=re.I)
        if not tokens or re.sub(r'[\s\-–—,，、]', '', remainder):
            continue
        # A label must start with a dimension, not a relation on its own.
        first = re.sub(r'\s+', ' ', tokens[0][0]).lower()
        if first not in DIMENSIONS and not re.match(r'^\s*(?:实体(?:/IP)?|人物|物品|场景|动作|镜头/运镜|特效|音色|音乐|旋律|音效|环境音|人声)\s*[-–—]', piece, re.I):
            continue
        for token in tokens:
            key = re.sub(r'\s+', ' ', token[0]).lower()
            edits.append({'start': match.start()+token.start(), 'end': match.start()+token.end(),
                          'old': token[0], 'new': _TOKENS[key]})
        for dash in re.finditer('[–—]', piece):
            edits.append({'start': match.start()+dash.start(), 'end': match.start()+dash.end(),
                          'old': dash[0], 'new': '-'})
    return sorted(edits, key=lambda op: (op['start'], op['end']))


def normalize(field, text):
    for op in reversed(operations(field, text)):
        text = text[:op['start']] + op['new'] + text[op['end']:]
    return text


def unknown_english(field, text):
    if field not in {'video_similarity_tag', 'similarity_tag', 'audio_similarity_tag', '视听标签'}:
        return False
    if re.search(r'(?m)^[ \t]*#{2,6}[ \t]*\S',text):return False
    translated = normalize(field, text)
    # IP is an established abbreviation; do not translate or flag it.
    return bool(re.search(r'[A-Za-z]{2,}', re.sub(r'(?<![A-Za-z])IP(?![A-Za-z])', '', translated, flags=re.I)))


def header(prefix, name):
    """Normalize only the recognized heading span; inline body is separate."""
    result = re.sub(r'^(\ufeff?["“]?)##(?!#)[ \t\u3000]*', r'\1## ', prefix)
    if name in TITLES:
        aliases = [name, TITLES[name], *(k for k, v in ALIASES.items() if v == name)]
        if name == 'video_similarity_tag': aliases.append('similarity_tag')
        pattern = '|'.join(re.escape(k) for k in sorted(set(aliases), key=len, reverse=True))
        result = re.sub(r'(?m)^(\ufeff?["“]?(?:##[ \t]*)?)(?:'+pattern+r')(?=[ \t:：\r\n]|$)',
                        lambda m: m[1]+TITLES[name], result)
    return result


def confirmation_reason(field, text):
    if field in {'video_similarity_tag','audio_similarity_tag','similarity_tag','视听标签'} and normalize(field,text).strip() in {'是','否'}:
        return '此处是视觉/听觉标签，却填写了“是/否”关系判断值；请核对字段归属，并补充对应的中文维度标签。'
    if unknown_english(field,text):
        return '标签中有未收录的英文值，无法可靠转换；请确认对应中文标签，原值保留。'
    return ''
