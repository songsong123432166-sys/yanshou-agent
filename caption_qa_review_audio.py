"""Audio protection shared by detector and review resolution; no I/O."""
import re


def audio_spans(text):
    """Only interior edits are protected: outer paragraph periods remain valid."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"[{｛][^{}｛｝]*(?:[}｝]|$)", text)]
    for match in re.finditer(r"[<＜][^<>＜＞]*(?:[>＞]|$)", text):
        if not re.fullmatch(r"[<＜]\s*(?:主体|场景|图片|视频|音频|画面元素)\s*\d+\s*[>＞]", match[0]):
            spans.append((match.start(), match.end()))
    stack = []
    for index, char in enumerate(text):
        if char in "（(":
            stack.append((char, index))
        elif char in "）)" and stack and (stack[-1][0], char) in {("（", "）"), ("(", ")")}:
            _, start = stack.pop()
            if re.search(r"音乐|配乐|伴奏|BGM|演唱|演奏|音效|环境音|人声|歌声|节奏|旋律", text[start:index + 1], re.I):
                spans.append((start, index + 1))
    for _, start in stack:
        if re.search(r"音乐|配乐|BGM|人声|音效|环境音", text[start:], re.I):
            spans.append((start, len(text)))
    offset = 0
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*(?:<主体\d+>\s*[：:]\s*)?(?:画外|旁白|背景音乐|配乐|环境音|BGM)", line, re.I):
            spans.append((offset, offset + len(line.rstrip())))
        offset += len(line)
    return spans


def touches_audio(old, start, end):
    return any((start < b and end > a) if start != end else a < start < b for a, b in audio_spans(old))
