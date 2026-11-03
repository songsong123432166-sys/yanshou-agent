"""Deterministic Caption policy snapshot. No model client, API key or network code."""
from __future__ import annotations
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from caption_qa_symbols import CONTENT_SYMBOLS, LITERAL_RE, SYMBOL_DELIMITERS, mask_literals, symbol_findings
from _time_layout import normalize as normalize_time_layout
from _spacing import normalize as normalize_spacing
import caption_qa_width as width
from _schema import ALIASES as FIELD_ALIASES, is_generation_request, preferred_header_indices

from _labels import separator_operations, normalize_separators

SECTION_ALIASES = {alias: name for alias, name in FIELD_ALIASES.items()
                   if name not in {"_task", "_subtask"} and alias != name}

PLAIN_SECTION_NAMES = ("任务类型", "提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令",
                       *SECTION_ALIASES, "画面元素", "similarity_label",
                       "video_similarity_tag", "audio_similarity_tag", "task_type", "subtask_type", "similarity_tag", "视听标签")

_PLAIN_HEADER_PATTERN = "|".join(re.escape(name) for name in sorted(PLAIN_SECTION_NAMES, key=len, reverse=True))

_HEADER_PREFIX = r'^\ufeff?["“]?'

SECTION_RE = re.compile(
    _HEADER_PREFIX + r'(?:##(?!#)[ \t]*)?(?P<inline_name>' + _PLAIN_HEADER_PATTERN
    + r')(?:[ \t]{2,}|[：:][ \t]*)(?=\S)|'
    + _HEADER_PREFIX + r'(?:##(?!#)[ \t]*(?P<markdown_name>[^\r\n]+)|[ \t]*(?P<plain_name>'
    + _PLAIN_HEADER_PATTERN + r')[ \t]*)\r?\n|'
    + _HEADER_PREFIX + r'(?:##(?!#)[ \t]*)?(?P<empty_name>' + _PLAIN_HEADER_PATTERN + r')[ \t]*\Z',
    re.MULTILINE)

SUBJECT_RE = re.compile(r"<主体\d+>")

BARE_RE = re.compile(r"(?<![<a-zA-Z0-9])(主体|视频|图片|音频|场景)\s*\d+(?![\d>])")

SPACED_TAG_RE = re.compile(r"<[ \t]*(主体|视频|图片|音频|场景)[ \t]*(\d+)[ \t]*>")

DUPLICATE_RE = re.compile(r"(?P<tag><(?:主体|场景|视频|图片)\d+>)(?P<sep>[：:\s]*)(?:(?P=tag)[：:\s]*)+")

LLM_TARGET_SECTIONS = {"提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}

ANCHOR_HEADERS = {"任务类型", "提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}

MULTI_LABEL_SECTIONS = frozenset({"任务类型", "video_similarity_tag", "audio_similarity_tag", "视听标签",
                                  "task_type", "subtask_type", "similarity_tag"})

PUNCTUATION_EXEMPT_SECTIONS = MULTI_LABEL_SECTIONS | {"similarity_label"}

AUTO_FIX_POLICY_VERSION = 9

SENTENCE_ENDINGS = "。！？!?"

class Issue(TypedDict):
    source: str
    section: str
    original_snippet: str
    issue_type: str
    suggestion: str
    can_auto_fix: bool
    resolved: bool

def _issue(source: str, section: str, snippet: str, kind: str,
           suggestion: str, auto: bool = False) -> Issue:
    return Issue(source=source, section=section, original_snippet=snippet,
                 issue_type=kind, suggestion=suggestion, can_auto_fix=auto, resolved=False)

def is_human_caption(text: Any) -> bool:
    """零 API 的人标格式判定；重复标题只计一次，单条 QA 入口不受该过滤影响。"""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False
    headers = {_section_name(match) for match in SECTION_RE.finditer(text)}
    anchors = headers & ANCHOR_HEADERS
    return len(anchors) >= 3 and bool(anchors & {"主体", "自然指令"})

def _raw_section_name(match):
    return (match.group("markdown_name") or match.group("plain_name")
            or match.group("inline_name") or match.group("empty_name")).strip()

def _section_name(match):
    name = _raw_section_name(match)
    return SECTION_ALIASES.get(name, name)

def _caption_headers(text):
    headers = []
    for match in SECTION_RE.finditer(text):
        name, start = _section_name(match), match.start()
        # Some exports repeat a plain label immediately after its Markdown
        # label. Preserve both, without treating an empty wrapper as content.
        if headers and headers[-1][0] == name and not text[headers[-1][2]:start].strip():
            _, start, _ = headers.pop()
        headers.append((name, start, match.end()))
    # Old exports sometimes put a generation prompt under subtask_type. Keep
    # its original title/spans, but check its body as prose rather than labels.
    resolved=[]
    for index,(name,start,body) in enumerate(headers):
        end=headers[index+1][1] if index+1<len(headers) else len(text)
        if name=='subtask_type' and is_generation_request(text[body:end]):
            name='提示'
        resolved.append((name,start,body))
    return resolved

def _preferred_caption_headers(caption,headers):
    entries=[{'field':name,'alias':_raw_section_name(SECTION_RE.match(caption,start)),
              'start':start,'end':body} for name,start,body in headers]
    return preferred_header_indices(caption,entries)

class CaptionSections(OrderedDict):
    """Active fields with informational source-version notes."""
    field_resolution_notes=()

def parse_caption_sections(caption: str) -> dict[str, str]:
    """按指定正则解析，保留内容空白及字段顺序；重复标题拒绝静默覆盖。

    兼容纯文本、Markdown、同行正文和空末尾标题；不改写原始标题。
    """
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Caption 必须是非空 Markdown 字符串")
    matches = _caption_headers(caption)
    if not matches or (caption[:matches[0][1]].strip() and not is_human_caption(caption)):
        raise ValueError("Caption 未能解析为支持的分段标题及正文；未完成内容检查")
    selected,notes=_preferred_caption_headers(caption,matches)
    sections=CaptionSections();sections.field_resolution_notes=notes
    for index, (name, start, body_start) in enumerate(matches):
        if index not in selected:continue
        if not name or name in sections:
            raise ValueError(f"字段名为空或重复：{name!r}")
        end = matches[index + 1][1] if index + 1 < len(matches) else len(caption)
        sections[name] = caption[body_start:end]
    # 未匹配的末尾标题不得吞入前一字段。
    if any(re.search(r"^##(?!#)", value, re.MULTILINE) for value in sections.values()):
        raise ValueError("存在无法解析的二级标题（检查末尾换行）")
    return sections

def rebuild_caption(original: str, sections: Mapping[str, str]) -> str:
    """只替换字段正文；标题空格、原始顺序、换行和前导空白原样保留。"""
    matches = _caption_headers(original)
    if not matches:
        return original
    selected,_=_preferred_caption_headers(original,matches)
    result = [original[:matches[0][1]]]
    for index, (name, start, body_start) in enumerate(matches):
        end = matches[index + 1][1] if index + 1 < len(matches) else len(original)
        body=sections.get(name,original[body_start:end]) if index in selected else original[body_start:end]
        result.extend((original[start:body_start],body))
    return "".join(result)

def _normalize_tags(text: str) -> str:
    pieces, cursor = [], 0
    for literal in LITERAL_RE.finditer(text):
        pieces.extend((_normalize_markup_tags(text[cursor:literal.start()]), literal[0]))
        cursor = literal.end()
    pieces.append(_normalize_markup_tags(text[cursor:]))
    return ''.join(pieces)

def _normalize_markup_tags(text: str) -> str:
    text = SPACED_TAG_RE.sub(lambda m: f"<{m[1]}{m[2]}>", text)
    return BARE_RE.sub(lambda m: "<" + re.sub(r"\s+", "", m.group()) + ">", text)

def _subject_definitions(text: str) -> list[str]:
    """Count declaration prefixes, never references inside descriptions."""
    view = DUPLICATE_RE.sub(lambda m: m.group("tag") + m.group("sep"), _normalize_tags(mask_literals(text)))
    definitions = []
    for line in view.splitlines():
        leading = re.match(r"^[ \t]*(<主体\d+>)", line)
        if leading:
            definitions.append(leading[1])
        # Also accept explicit same-line declarations after a sentence boundary.
        definitions.extend(re.findall(r"[。！？!?；;][ \t]*(<主体\d+>)[ \t]*[：:]", line))
    return definitions

def subject_coverage(sections: Mapping[str, str]) -> dict[str, list[str]]:
    # 格式缺失和连续前缀先规范化，避免将可修复格式误判为主体结构错误。
    defined = set(_subject_definitions(sections.get("主体", "")))
    summary_refs = set(SUBJECT_RE.findall(_normalize_tags(mask_literals(sections.get("总结", "")))))
    instruction_refs = set(SUBJECT_RE.findall(_normalize_tags(mask_literals(sections.get("自然指令", "")))))
    referenced = summary_refs | instruction_refs
    def ordered(tags: set[str]) -> list[str]:
        return sorted(tags, key=lambda tag: (int(re.search(r"\d+", tag)[0]), tag))  # type: ignore[index]
    return {"defined_subjects": ordered(defined), "referenced_subjects": ordered(referenced),
            "summary_referenced_subjects": ordered(summary_refs),
            "instruction_referenced_subjects": ordered(instruction_refs),
            "missing_in_instruction": ordered(defined - instruction_refs),
            "missing_subjects": ordered(defined - instruction_refs),
            "ghost_subjects": ordered(referenced - defined)}

def _source_static_rules(sections: Mapping[str, str]) -> list[Issue]:
    """纯 Python 静态预检；不猜测自然语言语义。每个非空行作为可检查段落。"""
    errors: list[Issue] = []
    coverage = subject_coverage(sections)
    subject_counts = Counter(_subject_definitions(sections.get("主体", "")))
    for tag, count in subject_counts.items():
        if count > 1:
            errors.append(_issue("static", "主体", tag, "duplicate_defined_subjects",
                f"【主体编号重复定义】在 ## 主体 中重复定义了 {tag}，请核对是否应顺延编号"))
    for tag in coverage["missing_subjects"]:
        errors.append(_issue("static", "自然指令", tag, "missing_subjects",
            f"【重点：自然指令漏写已定义主体】在 ## 主体 中已定义 {tag}，但 ## 自然指令 未引用该标签；"
            "总结中的引用不能替代自然指令。请人工核对，不自动补标签或动作。"))
    for tag in coverage["ghost_subjects"]:
        errors.append(_issue("static", "总结 / 自然指令", tag, "ghost_subjects",
            f"后文引用了未在 `## 主体` 中定义的标签 {tag}"))
    numbers = sorted({int(tag[3:-1]) for tag in coverage["defined_subjects"]})
    if numbers and any(number != index for index, number in enumerate(numbers, 1)):
        errors.append(_issue("static", "主体", "、".join(coverage["defined_subjects"]),
                             "skipped_numbering", "主体编号不连续（应从 1 开始）；请人工核对编号映射"))
    for name, value in sections.items():
        from _time_layout import operations as time_layout_operations
        for op in time_layout_operations(name,value):
            errors.append(_issue('static',name,op['heading'],'time_range_break',
                '新时间段另起一段，段间空一行；段内第几秒不强制换段。',True))
        if name in MULTI_LABEL_SECTIONS:
            label_lines = [line.strip() for line in value.splitlines() if line.strip()]
            embedded_header = re.search(r"(?m)^[ \t]*#{2,6}[ \t]*\S", value)
            if not embedded_header and (len(label_lines) > 1 or re.search(r"[、，,；;]|^[ \t]*[|｜]|[|｜][ \t]*$|[|｜][ \t]*[|｜]", value.strip())):
                empty_label = re.search(r"^[|｜]|[|｜]$|[|｜][ \t]*[|｜]", value.strip())
                message = ("多标签列表存在空标签：请检查首尾多余竖线或连续竖线，不自动改写标签" if empty_label else
                           "多个标签应使用 | 或 ｜ 拼接，不用顿号、逗号、分号或换行；实体/IP、镜头/运镜内部的 / 保留，请人工核对")
                safe=bool(separator_operations(name,value))
                if safe:message='标签完整，仅将分隔符规范为竖线；标签内容与顺序保留。'
                errors.append(_issue("static", name, value.strip(), "label_separator",message,safe))
        if name in LLM_TARGET_SECTIONS:
            view = mask_literals(value)
            for snippet, message in symbol_findings(value):
                errors.append(_issue("static", name, snippet, "symbol_format", message, False))
            for match in SPACED_TAG_RE.finditer(view):
                normalized = f"<{match[1]}{match[2]}>"
                if match.group() != normalized:
                    errors.append(_issue("static", name, match.group(), "bare_tags",
                                         "规范标签内部空格：" + normalized, True))
            for match in BARE_RE.finditer(view):
                if any(tag.start() <= match.start() < tag.end() for tag in SPACED_TAG_RE.finditer(view)):
                    continue
                errors.append(_issue("static", name, match.group(), "bare_tags",
                    "补齐尖括号：<" + re.sub(r"\s+", "", match.group()) + ">", True))
        if name not in PUNCTUATION_EXEMPT_SECTIONS:
            offset = 0
            literals = list(LITERAL_RE.finditer(value))
            for line in value.splitlines(keepends=True):
                text = line.strip()
                end = offset + len(line.rstrip())
                inside_literal = any(m.start() < end < m.end() or
                                     (end == m.end() and m[0][-1] not in '}】') for m in literals)
                if text and text[-1] not in SENTENCE_ENDINGS and not inside_literal:
                    errors.append(_issue("static", name, text, "missing_punctuation", "在段落末尾补全句号。", True))
                offset += len(line)
    return errors

def apply_static_fixes(sections):
    fixed=OrderedDict()
    for name,value in sections.items():
        value=normalize_spacing(name,normalize_separators(name,value))
        if name in PUNCTUATION_EXEMPT_SECTIONS:
            fixed[name]=value
            continue
        value=width.normalize_basic_width(name,value)
        if name in LLM_TARGET_SECTIONS:
            value=width.outside_literals(value,_normalize_markup_tags)
        value=normalize_time_layout(name,value)
        fixed[name]=width.finish_paragraphs(value,SENTENCE_ENDINGS)
    return fixed


def run_static_rules(sections):
    # Existing English full stops and ellipses are retained, not missing marks.
    return [item for item in _source_static_rules(sections)
            if not (item['issue_type']=='missing_punctuation' and
                    item['original_snippet'].rstrip().endswith(('.', '…')))]
