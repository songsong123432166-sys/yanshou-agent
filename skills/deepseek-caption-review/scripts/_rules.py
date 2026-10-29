"""Shared deterministic QA definitions; no API client, credentials or batch runner."""
from __future__ import annotations
import re
from collections import Counter
from collections.abc import Mapping
from typing import TypedDict
from caption_qa_symbols import LITERAL_RE, mask_literals, symbol_findings

SUBJECT_RE = re.compile(r"<主体\d+>")

BARE_RE = re.compile(r"(?<![<a-zA-Z0-9])(主体|视频|图片|音频|场景)\s*\d+(?![\d>])")

SPACED_TAG_RE = re.compile(r"<[ \t]*(主体|视频|图片|音频|场景)[ \t]*(\d+)[ \t]*>")

DUPLICATE_RE = re.compile(r"(?P<tag><(?:主体|场景|视频|图片)\d+>)(?P<sep>[：:\s]*)(?:(?P=tag)[：:\s]*)+")

LLM_TARGET_SECTIONS = {"提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}

MULTI_LABEL_SECTIONS = frozenset({"任务类型", "video_similarity_tag", "audio_similarity_tag", "视听标签",
                                  "task_type", "subtask_type", "similarity_tag"})

PUNCTUATION_EXEMPT_SECTIONS = MULTI_LABEL_SECTIONS | {"similarity_label"}

AUTO_FIX_POLICY_VERSION = 9

DISABLED_FINDING_RE = re.compile(
    r"属性冲突|跨字段.{0,8}(?:冲突|不一致)|(?:服饰|服装|颜色|材质|方位|属性).{0,16}(?:冲突|不一致|矛盾)"
    r"|长袍.{0,12}裙摆|裙摆.{0,12}长袍|叠词|衍字|重复(?:字|词|颜色)"
    r"|(?:字词|词语).{0,8}冗余|冗余.{0,8}(?:字词|词语)|重复标签|标签前缀")

UNAMBIGUOUS_TYPO_FIXES = {
    "眼晴": "眼睛", "帽孑": "帽子", "袖孑": "袖子", "鞋孑": "鞋子",
    "胳搏": "胳膊", "篮色": "蓝色", "帐蓬": "帐篷", "账篷": "帐篷",
}

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

def run_static_rules(sections: Mapping[str, str]) -> list[Issue]:
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
        if name in MULTI_LABEL_SECTIONS:
            label_lines = [line.strip() for line in value.splitlines() if line.strip()]
            embedded_header = re.search(r"(?m)^[ \t]*#{2,6}[ \t]*\S", value)
            if not embedded_header and (len(label_lines) > 1 or re.search(r"[、，,；;]|^[ \t]*[|｜]|[|｜][ \t]*$|[|｜][ \t]*[|｜]", value.strip())):
                empty_label = re.search(r"^[|｜]|[|｜]$|[|｜][ \t]*[|｜]", value.strip())
                message = ("多标签列表存在空标签：请检查首尾多余竖线或连续竖线，不自动改写标签" if empty_label else
                           "多个标签应使用 | 或 ｜ 拼接，不用顿号、逗号、分号或换行；实体/IP、镜头/运镜内部的 / 保留，请人工核对")
                errors.append(_issue("static", name, value.strip(), "label_separator",
                    message, False))
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
