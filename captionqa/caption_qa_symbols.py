"""Caption symbol contract; literal dialogue/on-screen text is opaque content."""
import re


CONTENT_SYMBOLS = {"主体": "<主体n>", "场景": "<场景n>", "人声": "{}",
                   "参考素材": "<图片X>/<视频X>/<音频X>", "多标签拼接": "| 或 ｜",
                   "音效": "<>", "环境音": "<>", "音乐": "（）", "文字": "【】"}
PAIRS = {"<": ">", "{": "}", "（": "）", "【": "】"}
SYMBOL_DELIMITERS = frozenset("".join(PAIRS) + "".join(PAIRS.values()))
LITERAL_RE = re.compile(r"\{[^{}]*(?:\}|$)|【[^【】]*(?:】|$)")


def mask_literals(text):
    """Preserve offsets/newlines while hiding spoken and displayed literal words."""
    return LITERAL_RE.sub(lambda m: ''.join(c if c in '\r\n' else ' ' for c in m[0]), text)


def symbol_findings(text):
    """Check paired prescribed delimiters, without inferring audio/visual meaning."""
    stack, findings = [], []
    closes = set(PAIRS.values())
    for pos, char in enumerate(text):
        # Foreign punctuation, comparisons and quoted symbols inside literal
        # speech/text are content, not additional Caption markup.
        if stack and stack[-1][0] in {'{', '【'}:
            opener = stack[-1][0]
            if char == PAIRS[opener]:
                stack.pop()
            elif char == opener:
                stack.append((char, pos))
            continue
        if char in PAIRS:
            stack.append((char, pos))
        elif char in closes:
            if stack and PAIRS[stack[-1][0]] == char:
                stack.pop()
            else:
                message = ("中文右括号 ） 缺少匹配的中文左括号（，或与英文括号混用；请人工核对" if char == '）' else
                           f"符号 {char} 缺少匹配的左符号或嵌套顺序不正确；请按内容符号规范核对")
                findings.append((pos, message))
    for char, pos in stack:
        findings.append((pos, f"符号 {char} 缺少匹配的 {PAIRS[char]}；请人工核对，不自动补齐"))
    return [(text[max(0, pos - 16):pos + 17], message) for pos, message in sorted(findings)]
