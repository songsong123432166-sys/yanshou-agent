#!/usr/bin/env python3
"""三阶段结构化 Caption 质检。Python 3.10+；dry-run 仅使用标准库。

运行：python agent.py check --dry-run
批量：python agent.py check --input data.jsonl --output result.jsonl --concurrency 10
库调用：CaptionQAAgent(dry_run=True).check(caption)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TextIO, TypedDict
from captionqa.caption_qa_symbols import CONTENT_SYMBOLS, LITERAL_RE, SYMBOL_DELIMITERS, mask_literals, symbol_findings
from captionqa.caption_field_schema import ALIASES as FIELD_ALIASES, is_generation_request, preferred_header_indices


_REQUEST_PROGRESS = ContextVar('caption_request_progress', default=None)


def _notify_request_progress(state: str, **details: Any) -> None:
    """Per-task telemetry; a UI callback must never change a QA outcome."""
    callback = _REQUEST_PROGRESS.get()
    if callback is not None:
        try:
            callback(dict(state=state, **details))
        except Exception:
            pass


LLM_SYSTEM_PROMPT = """
你是一个严谨的视频 Caption 文本质检专家。检查客观问题，严格按自动修复白名单执行最小修改，其他问题仅提示并转人工。

【严格遵守的质检规则】
1. 最小化修改原则（Minimal Edit）：严禁主观润色文风，严禁删减任何画面视觉细节（颜色、材质、服装、方位、动作）。只修改客观硬伤。
2. 自动修复严格限于以下三类，issue_type 使用对应的精确名称：
   - 裸标签漏写 <>：仅给已经明确写出的“主体1”等编号补齐尖括号，禁止凭自然语言指代新增实体标签。
   - 无歧义错别字：仅自动修复确定性纠错词表中的明确错字；词表随用户请求传入。未命中词表或存在歧义时 can_auto_fix=false，保留原文并转人工。若有明确且唯一的局部候选写法，可在该 issue 的可选 replacement 字段返回 original_snippet 对应的候选文本，供人工确认后执行；不得将未确认候选合入 corrected_sections。没有唯一改法时省略 replacement。不得借纠错改颜色、材质、方位、服饰或动作事实，不用此字段提供病句重写或补主体。
   - 标点符号硬伤：只增加、删除或替换标点，保留全部正文和原有空白。例如“依次做出、痛苦挣扎的动作”删去多余顿号；“眉头紧锁、身穿白色长袍”改为逗号。需要人工确认（包括对白内部）且有唯一明确的标点改法时，可用可选 replacement 返回 original_snippet 对应的完整候选；can_auto_fix=false，不合入 corrected_sections。不得修改台词文字、数值、时间码、网址或内容包裹符。
   - 句末标点规则：只有“任务类型”“similarity_label”“video_similarity_tag”“audio_similarity_tag”四类字段及其已知导出别名 task_type、subtask_type、similarity_tag、视听标签豁免；其他字段（包括提示、基调、画面元素）的每个非空段落都需要句末标点。“无”也需要写为“无。”。保留已有句号、问号和叹号。仅当 }、”、）等闭合符位于段尾，后面没有其他文字且没有句末标点时，才补外层句号。例如段尾“说道{私は異世界に来たようだ。}”补为“说道{私は異世界に来たようだ。}。”。句中“说道{你好}，然后转身。”保持原样，禁止因为出现 } 就在其后补句号。段尾对白内部的句号不能替代外层句末标点，不得把此处必需的外层句号反判为冗余。
3. 只提示/转人工的检查：
   - 内容符号不规范：参考素材用 <图片X>/<视频X>/<音频X>，适用于输入描述、场景、主体、总结、自然指令；场景和主体的定义与调用用 <场景X>/<主体X>；画面文字、花字、水印台标用【文字内容】，适用于输入描述、画面元素、自然指令；人物台词、演唱和喊叫用 {台词/人声}，背景音乐特征及变化用中文（背景音乐…），动作/环境音效或声音起止状态用 <音效描述>，后三类适用于自然指令（细节）。这些范围用于按语境核对，不因其他字段出现合法符号就直接判错。允许音乐（）内嵌人声{}及编号标签；音效/环境音的 <> 内允许整段描述。任务类型和视听标签多个标签用 | 或 ｜ 拼接；实体/IP、镜头/运镜里的 / 是标签本身，不拆分。对白中的韩语/日语等外语、可见文字的原内容必须保留。根据明确语境发现包裹符号用错时 can_auto_fix=false；缺少可靠语境不能凭某个关键词断定类别，普通括号、引号和外语标点本身不是错误，不猜测缺失内容，不自动换包裹符号。
   - 语法病句与语句不通顺：成分残缺、句式杂糅、语序颠倒、中途暗换主语等需要增删或重排正文的改写，一律 can_auto_fix=false。有明确、完整且保留原意的局部改稿时，在可选 replacement 中返回 original_snippet 对应的完整替换文本，供人工核对并确认；不合入 corrected_sections。只给修改方向、没有可靠改稿时省略 replacement。
   - 隐式漏挂主体标签：自然语言指代可能漏挂标签时，仅提示并转人工，不自动补标签、替换指代或补动作。有明确指代依据时，可用 replacement 给出局部标签引用候选供人工确认；不得编造主体、动作或画面事实。
   - 【重点：自然指令漏写已定义主体】在“主体”中已定义的每个 <主体N>，必须在“自然指令”中明确引用；只在“总结”中出现不能替代自然指令的引用。“他/她/三人”等自然语言指代不能当作已引用标签。逐个列出自然指令缺失的主体编号，仅提示并转人工，禁止自动补标签或编造动作。
   - 主体漏写、重号、幽灵主体、编号断层等保持人工复核。主体重号仅统计定义前缀；“剑斜背在<主体1>身后”“音色同<主体2>”是描述内引用，不是重复定义。缺失标签不在被检查字段原文中，不得伪造为该字段的 original_snippet；这类问题仅写在 human_review_reason 中，issues 可为空。
5. 绝对豁免项（严禁报错）：
   - 已停用：属性冲突（含跨字段属性冲突）、衍字/叠词冗余、重复标签前缀。这三项不检查、不提示、不转人工、不修复；不能换成“语法病句”“错别字”“标点”再次报告。原有叠词和重复前缀保持原样，不得删除任何重复字词或重复标签。长袍/裙摆、颜色/服饰等属性不一致不属于本次质检范围。
   - 严禁挑刺 `## 自然指令` 中的时间戳逻辑（如“0-4秒...第0秒...”完全合规，保持原样）。
   - 严禁因为 `<场景N>` 或 `## 画面元素` 未在后文出现而报错（只要求已定义的 `<主体N>` 在自然指令中出现）。
6. 修订边界与差异化回传：
   - 新版字段优先，旧别名仅用于补充缺少或留空的新版字段。字段优先级由程序处理，不能把旧字段内容混入已选定的新版字段。
   - read_only_sections 中的字段仅作引用上下文，不检查其正文、不生成该字段的修改。画面元素定义存在时，不能将自然指令中的对应引用判成未定义。没有收到某字段不代表原数据没有该字段，信息不足时不得据此断言缺失。
   - 人工确认候选与自动修复分开：所有仍启用的检查项，只要有明确具体的局部修改方案，都应提供可选 replacement（对应 original_snippet 的完整替换文本，最长4000字），让人工核对“修改前 → 修改后”后确认执行。仅给方向或依赖补充事实时省略；候选不进入 corrected_sections、不因有方案就设 can_auto_fix=true。保留事实、时间戳和参考编号，音频及画面文字正文仍保留；已停用项不生成候选。
   - caption_sections 已完成静态标签/标点规范化，static_errors 仅包含仍未解决的静态问题或必须保留的主体结构告警；不要把已规范化的字段反判为静态误报。所有判断基于当前字段正文。
   - 若某个已定义的 `<主体N>` 在自然指令中未出现，即使总结已出现，也必须将 `need_human_review` 设为 true，明确列出缺失编号；严禁脑补编造该主体的画面动作。
   - 只有上述三类且修复确定时才能 can_auto_fix=true；其他仍启用的检查项必须 can_auto_fix=false 并设 need_human_review=true。已停用三项不得写入 issues 或 human_review_reason，也不能影响 need_human_review。禁止把语法改写、属性调整伪装成错字或标点修复。
   - original_snippet 必须是字段中确实存在的最短定位片段；只有该片段发生了符合其类型的修改才视为已修复。
   - 在输出的 `corrected_sections` 字典中，**只返回实际发生了文字/标点/标签修改的字段**！如果某个字段完全没有问题，不要在 `corrected_sections` 中返回该字段。

请严格输出以下 JSON 格式：
{
  "issues": [
    {
      "section": "所在字段名",
      "original_snippet": "有问题的原文片段",
      "issue_type": "裸标签漏写 <> | 无歧义错别字 | 标点符号硬伤 | 语法病句与语句不通顺 | 隐式漏挂主体标签 | 内容符号不规范",
      "suggestion": "最小修改建议",
      "can_auto_fix": true
    }
  ],
  "need_human_review": false,
  "human_review_reason": "任何白名单以外的问题、修订不确定性或主体结构告警均在此说明；否则为空字符串",
  "corrected_sections": {
    "有修改的字段名": "完成最小化修复后的该字段完整文本（未修改的字段请勿放入此字典）"
  }
}
"""

SECTION_ALIASES = {alias: name for alias, name in FIELD_ALIASES.items()
                   if name not in {"_task", "_subtask"} and alias != name}
SECTION_ALIASES['task_type']='任务类型'
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
TAG_RE = re.compile(r"<(?:主体|场景|视频|图片|音频|画面元素)\d+>")
# 加数字边界，避免 \d+ 回溯把 <主体12> 的“主体1”误判为裸标签。
BARE_RE = re.compile(r"(?<![<a-zA-Z0-9])(主体|视频|图片|音频|场景)\s*\d+(?![\d>])")
SPACED_TAG_RE = re.compile(r"<[ \t]*(主体|视频|图片|音频|场景)[ \t]*(\d+)[ \t]*>")
# 将标签和分隔符分开捕获：兼容冒号/空白不同、连续三次及无分隔符。
DUPLICATE_RE = re.compile(r"(?P<tag><(?:主体|场景|视频|图片)\d+>)(?P<sep>[：:\s]*)(?:(?P=tag)[：:\s]*)+")
TIMESTAMP_RE = re.compile(r"(?:第\s*)?\d+(?:\.\d+)?(?:\s*[-~～—至]\s*\d+(?:\.\d+)?)?\s*(?:秒|s\b)|\b\d{1,2}:\d{2}(?:\.\d+)?\b")
LLM_TARGET_SECTIONS = {"提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}
LLM_CONTEXT_SECTIONS = LLM_TARGET_SECTIONS | {"画面元素"}
ANCHOR_HEADERS = {"任务类型", "提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}
CAPTION_SCOPE_VERSION = 3
CAPTION_PARSER_VERSION = 5
QA_RESULT_COLUMNS = ("qa_status", "qa_missing_or_dup_subjects", "qa_issues_summary",
                     "qa_human_review_reason", "qa_revised_caption")
CORE_SECTIONS = LLM_TARGET_SECTIONS  # 兼容旧调用方；实际顺序始终取自原始 sections。
MULTI_LABEL_SECTIONS = frozenset({"任务类型", "video_similarity_tag", "audio_similarity_tag", "视听标签",
                                  "task_type", "subtask_type", "similarity_tag"})
PUNCTUATION_EXEMPT_SECTIONS = MULTI_LABEL_SECTIONS | {"similarity_label"}
HARD_TYPES = {"missing_subjects", "ghost_subjects", "skipped_numbering", "duplicate_defined_subjects"}
AUTO_FIX_POLICY_VERSION = 9
AUTO_LLM_ISSUE_TYPES = {"裸标签漏写 <>", "无歧义错别字", "标点符号硬伤"}
DISABLED_LLM_ISSUE_TYPES = {"重复标签前缀", "衍字/叠词冗余", "跨字段属性冲突", "属性冲突", "duplicate_prefixes"}
DISABLED_FINDING_RE = re.compile(
    r"属性冲突|跨字段.{0,8}(?:冲突|不一致)|(?:服饰|服装|颜色|材质|方位|属性).{0,16}(?:冲突|不一致|矛盾)"
    r"|长袍.{0,12}裙摆|裙摆.{0,12}长袍|叠词|衍字|重复(?:字|词|颜色)"
    r"|(?:字词|词语).{0,8}冗余|冗余.{0,8}(?:字词|词语)|重复标签|标签前缀")
ENABLED_REVIEW_REASON_RE = re.compile(r"语法|病句|成分残缺|语序|句式|主语|漏挂|漏写|未引用|幽灵主体|编号|重号|标点|句号|错别字|错字|尖括号|裸标签")
# 旧响应类型保留解析兼容性；仍须通过相同的逐处修改校验。
LEGACY_AUTO_ISSUE_TYPES = {"错别字与衍字", "标点与标签格式"}
REVIEW_LLM_ISSUE_TYPES = {"语法病句与语句不通顺", "隐式漏挂主体标签", "内容符号不规范"}
LLM_ISSUE_TYPES = AUTO_LLM_ISSUE_TYPES | LEGACY_AUTO_ISSUE_TYPES | REVIEW_LLM_ISSUE_TYPES
UNAMBIGUOUS_TYPO_FIXES = {
    "眼晴": "眼睛", "帽孑": "帽子", "袖孑": "袖子", "鞋孑": "鞋子",
    "胳搏": "胳膊", "篮色": "蓝色", "帐蓬": "帐篷", "账篷": "帐篷",
}
# 尖括号不在标点白名单中，实体格式由静态规则处理。
EDITABLE_PUNCTUATION_RE = re.compile(r'''[，。！？、；：,.!?;:“”‘’"'（）()【】\[\]…—-]''')
SENTENCE_ENDINGS = "。！？!?"


class Issue(TypedDict):
    source: str
    section: str
    original_snippet: str
    issue_type: str
    suggestion: str
    can_auto_fix: bool
    resolved: bool


class SemanticResult(TypedDict):
    issues: list[dict[str, Any]]
    need_human_review: bool
    human_review_reason: str
    corrected_sections: dict[str, str]
    performed: bool


class QAReport(TypedDict):
    caption_parser_version: int
    status: str
    qa_scope: str
    semantic_qa_performed: bool
    need_human_review: bool
    human_review_reason: str
    subject_check: dict[str, list[str]]
    static_errors: list[Issue]
    llm_issues: list[dict[str, Any]]
    issues: list[Issue]
    remaining_static_errors: list[Issue]
    revised_caption: str
    field_resolution_notes: list[dict[str, Any]]


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


def uses_legacy_instruction_heading(text):
    """Detect an actual legacy heading, not the phrase inside ordinary prose."""
    return isinstance(text,str) and any(name=='自然指令' and any(alias in text[start:body]
        for alias in ('自然描述','natural_instruction'))
        for name,start,body in _caption_headers(text))


def instruction_parser_outdated(text, version=1, *, semantic=False):
    """Invalidate only titles that the saved parser could not understand."""
    if not isinstance(text,str):return False
    headers=_caption_headers(text)
    names=[name for name,_,_ in headers]
    for index,(name,start,body) in enumerate(headers):
        raw=_raw_section_name(SECTION_RE.match(text,start))
        minimum={'自然描述':2,'natural_instruction':3,'subject':4,'task_type':5,
                 'brief_captions':4,'similarity_tag':4}.get(raw,1)
        if name=='提示' and raw=='subtask_type':minimum=4
        if names.count(name)>1 and raw==name and any(
                other==name and _raw_section_name(SECTION_RE.match(text,a))!=name
                for other,a,_ in headers):minimum=5
        if semantic and name=='画面元素' and re.search(r'<画面元素\d+>',text):minimum=5
        if (version or 1)<minimum:return True
    return False


def _preferred_caption_headers(caption,headers):
    entries=[{'field':name,'alias':_raw_section_name(SECTION_RE.match(caption,start)),
              'start':start,'end':body} for name,start,body in headers]
    return preferred_header_indices(caption,entries)


class CaptionSections(OrderedDict):
    """Active fields with informational source-version notes."""
    field_resolution_notes=()


def _nonempty_cell(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value.strip()) if isinstance(value, str) else True


def _excluded_caption_column(name: Any) -> bool:
    label = re.sub(r"\s+", "", name).lower() if isinstance(name, str) else ""
    return label.startswith("qa_") or any(word in label for word in
        ("机标", "机器", "machine", "模型", "备注", "note", "comment", "问题说明", "复核原因", "修改建议", "审核意见", "质检"))


def _caption_column_priority(name: Any) -> int:
    if not isinstance(name, str) or len(name) > 100 or "\n" in name or "\r" in name:
        return 0
    label = re.sub(r"\s+", "", name).lower()
    if _excluded_caption_column(name):
        return 0
    if "人标" in label or label in {"人工", "人工标注", "人工审核文本"}:
        return 3
    if "caption" in label:
        return 2
    return 0


def detect_caption_column(data: str | Path | Iterable[Mapping[str, Any]], *,
                          sample_size: int = 50, sheet_name: str | None = None) -> str:
    """只在可确认的 Caption/人标列中识别；多个候选要求明确选列。

    Excel 和 CSV 支持第一行分组、第二行 caption 的表头。备注和机标列不参与
    识别；没有明确列名时要求手动指定，不靠正文猜测其他列的用途。
    """
    if sample_size < 1:
        raise ValueError("sample_size 必须为正整数")
    if isinstance(data, (str, Path)):
        records = (record for _, record, _ in _indexed_input_records(Path(data), sheet_name=sheet_name))
    else:
        records = iter(data)
    counts: dict[str, int] = {}
    labels: dict[str, Any] = {}
    secondary_header = isinstance(data, (str, Path)) and Path(data).suffix.lower() in {".xlsx", ".csv"}
    sampled = 0
    try:
        for row in records:
            if not isinstance(row, Mapping) or not any(_nonempty_cell(v) for v in row.values()):
                continue
            sampled += 1
            for name, value in row.items():
                counts.setdefault(name, 0)
                labels.setdefault(name, name)
                if secondary_header and sampled == 1 and not _excluded_caption_column(name) and _caption_column_priority(value):
                    # Only the first data row can be the secondary header.
                    # Caption text itself contains newlines and cannot qualify.
                    labels[name] = value
                if _caption_column_priority(labels[name]) and is_human_caption(value):
                    counts[name] += 1
            if sampled >= sample_size and any(counts.values()):
                break
    finally:
        close = getattr(records, "close", None)
        if close is not None:
            close()
    candidates = [name for name, count in counts.items() if count]
    if not candidates:
        raise ValueError("未发现可确认的人标 Caption 列；请明确指定 Caption 列。不会检查备注、机标或其他列。未调用 API。")
    priority = max(_caption_column_priority(labels[name]) for name in candidates)
    candidates = [name for name in candidates if _caption_column_priority(labels[name]) == priority]
    if len(candidates) > 1:
        raise ValueError(f"发现多个 Caption 列 {candidates!r}，请明确指定一个列名。未调用 API。")
    column = candidates[0]
    print(f'[Auto-Detect] 成功锁定人标 Caption 所在列: "{column}" (有效人标行数: {counts[column]} / 总行数: {sampled})')
    return column


def _select_human_caption(record: Any, target_col: str) -> tuple[str | None, str]:
    row = _record_dict(record, target_col)
    if is_human_caption(row.get(target_col)):
        return target_col, row[target_col]
    return None, ""


def _skipped_report() -> QAReport:
    return QAReport(status="SKIPPED_NOT_HUMAN_CAPTION", qa_scope="human_caption_filter",
        semantic_qa_performed=False, need_human_review=False, human_review_reason="", subject_check={},
        static_errors=[], llm_issues=[], issues=[], remaining_static_errors=[], revised_caption="")


def _qa_result_values(report: QAReport) -> dict[str, str]:
    labels = {"missing_subjects": "自然指令漏写已定义主体", "ghost_subjects": "幽灵主体",
              "duplicate_defined_subjects": "主体重号", "skipped_numbering": "编号断层",
              "duplicate_prefixes": "重复标签", "bare_tags": "裸标签", "missing_punctuation": "句末标点缺失"}
    subjects = [f'{labels[i["issue_type"]]}: {i["original_snippet"]}' for i in report["static_errors"]
                if i["issue_type"] in {"missing_subjects", "ghost_subjects", "duplicate_defined_subjects"}]
    summaries = [f'{number}. [{i["section"]}-{labels.get(i["issue_type"], i["issue_type"])}] '
                 f'"{i["original_snippet"]}" -> {i["suggestion"]}' for number, i in enumerate(report["issues"], 1)]
    return {"qa_status": report["status"], "qa_missing_or_dup_subjects": "；".join(subjects),
            "qa_issues_summary": "；".join(summaries), "qa_human_review_reason": report["human_review_reason"],
            "qa_revised_caption": report["revised_caption"] if report["status"] in {"AUTO_FIXED", "NEED_HUMAN_REVIEW"} else ""}


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


def apply_static_fixes(sections: Mapping[str, str]) -> dict[str, str]:
    """确定性修复：裸标签、缺失句末标点；重复前缀保持原样。"""
    fixed: dict[str, str] = OrderedDict()
    for name, value in sections.items():
        if name in LLM_TARGET_SECTIONS:
            value = _normalize_tags(value)
        if name not in PUNCTUATION_EXEMPT_SECTIONS:
            lines: list[str] = []
            offset = 0
            literals = list(LITERAL_RE.finditer(value))
            for line in value.splitlines(keepends=True):
                text = line.rstrip()
                end = offset + len(text)
                inside_literal = any(m.start() < end < m.end() or
                                     (end == m.end() and m[0][-1] not in '}】') for m in literals)
                original_length = len(line)
                if text.strip() and text[-1] not in SENTENCE_ENDINGS and not inside_literal:
                    # 英文/中文逗号等已有尾符属于标点误用，交给语义阶段；此处只补句号。
                    line = text + "。" + line[len(text):]
                lines.append(line)
                offset += original_length
            value = "".join(lines)
        fixed[name] = value
    return fixed


def _disabled_finding(item: Mapping[str, Any], sections: Mapping[str, str], proposed: Mapping[str, str]) -> bool:
    if item["issue_type"] in DISABLED_LLM_ISSUE_TYPES or DISABLED_FINDING_RE.search(item["suggestion"]):
        return True
    # 兼容旧混合类型，同时阻止把纯重复删除包装成错字或语法修复。
    name = item["section"]
    if name in sections and name in proposed and isinstance(proposed[name], str):
        old, new = sections[name].strip(), proposed[name].strip()
        if old != new and DUPLICATE_RE.sub(lambda m: m.group("tag") + m.group("sep"), old) == new:
            return True
        changes = [(op, a, b, c, d) for op, a, b, c, d in
                   difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op != "equal"]
        if changes and all(op == "delete" and _is_redundant_deletion(old, new, a, b, c, d)
                           for op, a, b, c, d in changes):
            return True
    return False


def _validate_semantic(data: Any, sections: Mapping[str, str]) -> SemanticResult:
    """json_object 不保证 schema，因此逐项校验而非信任模型输出。"""
    if not isinstance(data, dict):
        raise ValueError("LLM JSON 顶层必须是对象")
    for key, expected in (("issues", list), ("need_human_review", bool),
                          ("human_review_reason", str), ("corrected_sections", dict)):
        if type(data.get(key)) is not expected:
            raise ValueError(f"LLM JSON 字段 {key} 类型错误或缺失")
    ignored, retained = [], []
    for item in data["issues"]:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in
            ("section", "original_snippet", "issue_type", "suggestion")) or type(item.get("can_auto_fix")) is not bool:
            raise ValueError("LLM issues 元素结构不合法")
        if _disabled_finding(item, sections, data["corrected_sections"]):
            ignored.append(item)
            continue
        if item["section"] not in LLM_TARGET_SECTIONS or item["section"] not in sections or item["issue_type"] not in LLM_ISSUE_TYPES:
            raise ValueError("LLM issue 字段名或问题类型不合法")
        if not item["original_snippet"] or item["original_snippet"] not in sections[item["section"]]:
            raise ValueError("LLM 问题片段不在原字段中")
        if 'replacement' in item and (not isinstance(item['replacement'],str) or len(item['replacement'])>4000):
            raise ValueError('LLM 候选替换必须是4000字以内的文本')
        retained.append(item)
    for name, value in data["corrected_sections"].items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("corrected_sections 必须是字符串字段映射")
        if name not in LLM_TARGET_SECTIONS or name not in sections:
            raise ValueError(f"LLM 试图新增或修改非白名单字段：{name}")
    # 模型声明 can_auto_fix=true 不能扩大程序的自动修复范围。
    issues = [dict(item, can_auto_fix=item["can_auto_fix"] and
                   item["issue_type"] in AUTO_LLM_ISSUE_TYPES | LEGACY_AUTO_ISSUE_TYPES)
              for item in retained]
    parts = re.split(r"[；;。\n，,]+", data["human_review_reason"])
    removed = bool(ignored) or any(DISABLED_FINDING_RE.search(part) for part in parts)
    corrected = {}
    for name, candidate in data["corrected_sections"].items():
        enabled = [i for i in issues if i["section"] == name]
        disabled = [i for i in ignored if i["section"] == name]
        if removed and not enabled:
            continue  # 不能让停用项目的改稿触发 unsafe_llm_edit，间接进入问题名单。
        if disabled:
            old, new = sections[name].strip(), candidate.strip()
            spans = [(m.start(), m.end()) for i in disabled if i["original_snippet"]
                     for m in re.finditer(re.escape(i["original_snippet"]), old)]
            pieces = []
            for op, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
                inside_disabled = any(start <= a and b <= end for start, end in spans)
                if op != "equal" and inside_disabled and not _whitelisted_changes(old, old[:a] + new[c:d] + old[b:], enabled):
                    pieces.append(old[a:b])
                else:
                    pieces.append(new[c:d])
            candidate = _pad_like(sections[name], "".join(pieces))
        if candidate.strip() != sections[name].strip():
            corrected[name] = candidate
    reason = "；".join(part.strip() for part in parts if part.strip() and not DISABLED_FINDING_RE.search(part)
                      and (not ignored or ENABLED_REVIEW_REASON_RE.search(part)))
    review = data["need_human_review"] and (bool(reason) or not removed)
    return SemanticResult(issues=issues, need_human_review=review,
        human_review_reason=reason, corrected_sections=corrected, performed=True)


def _semantic_fallback(reason: str) -> SemanticResult:
    return SemanticResult(issues=[], need_human_review=True, human_review_reason=reason,
                          corrected_sections={}, performed=False)


def _empty_semantic() -> SemanticResult:
    return SemanticResult(issues=[], need_human_review=False, human_review_reason="",
                          corrected_sections={}, performed=False)


def _client_options(api_key: str | None, base_url: str | None, timeout: float) -> dict[str, Any]:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("未设置 OPENAI_API_KEY，语义质检未执行")
    options: dict[str, Any] = dict(api_key=key, timeout=timeout, max_retries=0)
    if base_url or os.getenv("OPENAI_BASE_URL"):
        options["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
    return options


def _request_options(sections: Mapping[str, str], static_errors: Sequence[Issue],
                     model_name: str | None, timeout: float) -> dict[str, Any]:
    # 单次调用只构造这一条 Caption；非白名单字段及其错误片段均不外发。
    remaining = {(i["section"], i["issue_type"], i["original_snippet"]) for i in run_static_rules(sections)}
    pending = [i for i in static_errors if i["issue_type"] in HARD_TYPES or
               (i["section"], i["issue_type"], i["original_snippet"]) in remaining]
    payload = {"caption_sections": {k: v.strip() for k, v in sections.items() if k in LLM_CONTEXT_SECTIONS},
               "read_only_sections": sorted(LLM_CONTEXT_SECTIONS-LLM_TARGET_SECTIONS),
               "checked_sections": sorted(LLM_TARGET_SECTIONS),
               "static_errors": [i for i in pending if i["section"] in LLM_TARGET_SECTIONS or i["issue_type"] in HARD_TYPES],
               "auto_fix_policy": {"version": AUTO_FIX_POLICY_VERSION,
                   "allowed_issue_types": sorted(AUTO_LLM_ISSUE_TYPES),
                   "disabled_issue_types": sorted(DISABLED_LLM_ISSUE_TYPES),
                   "unambiguous_typo_fixes": UNAMBIGUOUS_TYPO_FIXES,
                   "punctuation_exempt_sections": sorted(PUNCTUATION_EXEMPT_SECTIONS),
                   "sentence_endings": SENTENCE_ENDINGS,
                   "require_outer_sentence_ending": True,
                   "outer_sentence_ending_scope": "paragraph_end_only",
                   "add_period_after_midparagraph_brace": False,
                   "content_symbols": CONTENT_SYMBOLS,
                   "symbol_category_changes_require_review": True,
                   "preserve_literal_speech_and_screen_text": True,
                   "require_defined_subjects_in_natural_instruction": True},
               "subject_coverage": subject_coverage(sections)}
    return dict(model=model_name or os.getenv("MODEL_NAME") or "gpt-4o-mini",
                messages=[{"role": "system", "content": LLM_SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"}, timeout=timeout)


def _parse_llm_response(response: Any, sections: Mapping[str, str]) -> SemanticResult:
    if not getattr(response, "choices", None):
        raise ValueError("LLM 未返回候选结果")
    choice = response.choices[0]
    if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
        raise ValueError("LLM 输出被截断、拒绝或过滤")
    if not isinstance(choice.message.content, str):
        raise ValueError("LLM 未返回文本 JSON")
    return _validate_semantic(json.loads(choice.message.content), sections)


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or 500 <= status < 600
    return isinstance(exc, (ValueError, TimeoutError, ConnectionError, OSError)) or type(exc).__name__ in {
        "RateLimitError", "APITimeoutError", "APIConnectionError"}


def _valid_llm_parameters(max_retries: int, timeout: float, retry_delay: float) -> bool:
    return (isinstance(max_retries, int) and 0 <= max_retries <= 3 and math.isfinite(timeout) and timeout > 0
            and math.isfinite(retry_delay) and retry_delay >= 0)


def run_llm_semantic_qa(sections: Mapping[str, str], static_errors: Sequence[Issue], *,
                        client: Any = None, model_name: str | None = None,
                        api_key: str | None = None, base_url: str | None = None,
                        max_retries: int = 3, timeout: float = 60.0,
                        retry_delay: float = 1.0) -> SemanticResult:
    """Chat Completions + JSON 模式；最多 1 + max_retries 次调用。

    注入 client 便于离线测试；SDK 自带重试关闭以免双重重试。
    失败返回 performed=False 的人工复核结果，不把失败误判成 PASS。
    """
    if not _valid_llm_parameters(max_retries, timeout, retry_delay):
        return _semantic_fallback("LLM 参数非法：额外重试须在 0-3 之间，超时为正数，退避时间非负")
    owned = client is None
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(**_client_options(api_key, base_url, timeout))
        except ImportError:
            return _semantic_fallback("未安装 openai SDK；请执行 python -m pip install openai")
        except ValueError:
            return _semantic_fallback("API 客户端配置无效或未设置 OPENAI_API_KEY")
        except Exception as exc:
            return _semantic_fallback(f"LLM 客户端初始化失败（{type(exc).__name__}）")
    try:
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(**_request_options(sections, static_errors, model_name, timeout))
                return _parse_llm_response(response, sections)
            except Exception as exc:
                reason = f"LLM 请求或 JSON 校验失败（{type(exc).__name__}；已尝试 {attempt + 1} 次），需人工复核"
                if not _retryable(exc) or attempt == max_retries:
                    return _semantic_fallback(reason)
                time.sleep(min(retry_delay * (2 ** attempt), 10.0))
        return _semantic_fallback("LLM 质检未完成")
    finally:
        if owned:
            client.close()


async def arun_llm_semantic_qa(sections: Mapping[str, str], static_errors: Sequence[Issue], *,
                             client: Any = None, model_name: str | None = None,
                             api_key: str | None = None, base_url: str | None = None,
                             max_retries: int = 3, timeout: float = 60.0,
                             retry_delay: float = 1.0, total_timeout: float | None = None) -> SemanticResult:
    """真正的异步网络调用，退避使用 asyncio.sleep；取消任务会向上传播。"""
    if not _valid_llm_parameters(max_retries, timeout, retry_delay):
        return _semantic_fallback("LLM 参数非法：额外重试须在 0-3 之间，超时为正数，退避时间非负")
    if total_timeout is not None and (not math.isfinite(total_timeout) or total_timeout <= 0):
        return _semantic_fallback('单条 AI 检查总时限必须为有限正数')
    owned = client is None
    if owned:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(**_client_options(api_key, base_url, timeout))
        except ImportError:
            return _semantic_fallback("未安装 openai SDK；请执行 python -m pip install openai")
        except ValueError:
            return _semantic_fallback("API 客户端配置无效或未设置 OPENAI_API_KEY")
        except Exception as exc:
            return _semantic_fallback(f"LLM 客户端初始化失败（{type(exc).__name__}）")
    attempts = 0

    async def request_with_retries():
        nonlocal attempts
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            _notify_request_progress('requesting', attempt=attempts, max_attempts=max_retries + 1)
            try:
                response = await asyncio.wait_for(client.chat.completions.create(
                    **_request_options(sections, static_errors, model_name, timeout)), timeout=timeout)
                return _parse_llm_response(response, sections)
            except Exception as exc:
                reason = f"LLM 请求或 JSON 校验失败（{type(exc).__name__}；已尝试 {attempt + 1} 次），需人工复核"
                if not _retryable(exc) or attempt == max_retries:
                    return _semantic_fallback(reason)
                delay = min(retry_delay * (2 ** attempt), 10.0)
                _notify_request_progress('retry_wait', attempt=attempts, max_attempts=max_retries + 1,
                                         retry_in=delay, error_type=type(exc).__name__)
                await asyncio.sleep(delay)
        return _semantic_fallback("LLM 质检未完成")

    try:
        if total_timeout is None:
            return await request_with_retries()
        try:
            return await asyncio.wait_for(request_with_retries(), timeout=total_timeout)
        except asyncio.TimeoutError:
            _notify_request_progress('timed_out', attempt=attempts, max_attempts=max_retries + 1)
            return _semantic_fallback(f'单条 AI 检查超过总时限 {total_timeout:g} 秒（已尝试 {attempts} 次），'
                                      '检测未完成；本地检查结果保留，可单独重试')
    finally:
        if owned:
            await client.close()


def _pad_like(original: str, candidate: str) -> str:
    """恢复模型通常会去掉的字段首尾空白。"""
    if not original.strip():
        return candidate
    prefix = original[:len(original) - len(original.lstrip())]
    suffix = original[len(original.rstrip()):]
    content = candidate.strip()
    if "\r\n" in original:
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    return prefix + content + suffix


def _changes_supported(old: str, new: str, issues: Sequence[dict[str, Any]]) -> bool:
    """每处差异必须落在可自动修复的原文片段中，禁止无依据整段改写。"""
    spans: list[tuple[int, int]] = []
    for item in issues:
        snippet = item["original_snippet"]
        if item["can_auto_fix"] and snippet:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(snippet), old))
    return all(any(a <= i1 and i2 <= b for a, b in spans)
               for op, i1, i2, _, _ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
               if op != "equal")


def _typo_edit_operations(old: str) -> set[tuple[str, int, int, str]]:
    """只有命中明确纠错词表的具体替换操作才能自动应用。"""
    allowed: set[tuple[str, int, int, str]] = set()
    for wrong, right in UNAMBIGUOUS_TYPO_FIXES.items():
        for match in re.finditer(re.escape(wrong), old):
            for op, a, b, c, d in difflib.SequenceMatcher(None, wrong, right, autojunk=False).get_opcodes():
                if op != "equal":
                    allowed.add((op, match.start() + a, match.start() + b, right[c:d]))
    return allowed


def _is_redundant_deletion(old: str, new: str, a: int, b: int, c: int, d: int) -> bool:
    removed = old[a:b]
    if c != d or not re.fullmatch(r"[\u3400-\u9fff]+", removed):
        return False
    for width in range(1, min(4, len(removed)) + 1):
        unit = removed[:width]
        if len(removed) % width or unit * (len(removed) // width) != removed:
            continue
        if old[max(0, a - width):a] != unit and old[b:b + width] != unit:
            continue
        # 保留正常单字叠词；除功能词衍字外，至少留下两个字。
        if width == 1 and unit not in "的地得了着过在是和与":
            left = len(new[:c]) - len(new[:c].rstrip(unit))
            right = len(new[c:]) - len(new[c:].lstrip(unit))
            if left + right < 2:
                continue
        return True
    return False


def _whitelisted_changes(old: str, new: str, issues: Sequence[dict[str, Any]]) -> bool:
    """逐处验证修改类型和实际动作，禁止用大范围问题片段授权正文改写。"""
    spans: list[tuple[int, int, str]] = []
    for item in issues:
        if item["can_auto_fix"] and item["issue_type"] in AUTO_LLM_ISSUE_TYPES | LEGACY_AUTO_ISSUE_TYPES:
            spans.extend((m.start(), m.end(), item["issue_type"])
                         for m in re.finditer(re.escape(item["original_snippet"]), old)
                         if item["original_snippet"])
    typo_ops = _typo_edit_operations(old)
    for op, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if any(char in SYMBOL_DELIMITERS for char in old[a:b] + new[c:d]):
            return False  # 内容类别符号只转人工，不允许伪装成普通标点修复。
        if any((a < span.end() and b > span.start()) or
               (a == b and span.start() < a <= span.end() and (a < span.end() or span[0][-1] not in '}】'))
               for span in LITERAL_RE.finditer(old)):
            return False  # 对白、演唱及画面文字保留原内容。
        kinds = {kind for start, end, kind in spans if start <= a and b <= end}
        punctuation_only = (EDITABLE_PUNCTUATION_RE.sub("", old[a:b]) ==
                            EDITABLE_PUNCTUATION_RE.sub("", new[c:d]))
        if ("。" in new[c:d] and old[:a].rstrip(" \t").endswith("}")
                and old[b:].split("\n", 1)[0].strip()):
            return False  # 句中闭合大括号后的内容未结束，不能因闭合符补句号。
        if kinds & {"标点符号硬伤", "标点与标签格式"} and punctuation_only:
            continue
        if kinds & {"无歧义错别字", "错别字与衍字"} and (op, a, b, new[c:d]) in typo_ops:
            continue
        return False
    return True


def _snippet_was_edited(old: str, new: str, snippet: str) -> bool:
    """插入标签可保留原名词；检查片段所在位置的实际差异，而非字符串消失。"""
    occurrences = list(re.finditer(re.escape(snippet), old))
    changes = [(i1, i2) for op, i1, i2, _, _ in
               difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op != "equal"]
    return bool(occurrences) and all(any(
        (m.start() <= a <= m.end() if a == b else a < m.end() and b > m.start())
        for a, b in changes) for m in occurrences)


class CaptionQAAgent:
    """单条入口 check；批量入口 check_batch。配置优先级：参数 > 环境变量 > 默认值。"""
    def __init__(self, *, dry_run: bool = False, client: Any = None, async_client: Any = None,
                 model_name: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_retries: int = 3,
                 timeout: float = 60.0, retry_delay: float = 1.0,
                 total_timeout: float = 180.0) -> None:
        if not _valid_llm_parameters(max_retries, timeout, retry_delay):
            raise ValueError("额外重试须在 0-3 之间，超时为有限正数，退避时间非负")
        if not math.isfinite(total_timeout) or total_timeout <= 0:
            raise ValueError('单条 AI 检查总时限必须为有限正数')
        self.total_timeout = total_timeout
        self.dry_run = dry_run
        self.async_client = async_client
        self._owns_async_client = False
        self.llm_options: dict[str, Any] = dict(client=client, model_name=model_name,
            api_key=api_key, base_url=base_url, max_retries=max_retries,
            timeout=timeout, retry_delay=retry_delay)

    def check(self, caption: str) -> QAReport:
        try:
            sections = parse_caption_sections(caption)
        except ValueError as exc:
            return self.failure_report(caption, str(exc))
        static_errors = run_static_rules(sections)
        cleaned = apply_static_fixes(sections)
        if self.dry_run:
            semantic = _empty_semantic()
        else:
            semantic = run_llm_semantic_qa(cleaned, static_errors, **self.llm_options)
        return self._assemble(caption, sections, cleaned, static_errors, semantic)

    async def acheck_single_caption(self, caption: str) -> QAReport:
        """1 Request = 1 Caption；批处理中共享同一 AsyncOpenAI 连接池。"""
        try:
            sections = parse_caption_sections(caption)
        except ValueError as exc:
            return self.failure_report(caption, str(exc))
        static_errors = run_static_rules(sections)
        cleaned = apply_static_fixes(sections)
        semantic = _empty_semantic()
        if not self.dry_run:
            if self.async_client is None:
                try:
                    from openai import AsyncOpenAI
                    self.async_client = AsyncOpenAI(**_client_options(
                        self.llm_options["api_key"], self.llm_options["base_url"], self.llm_options["timeout"]))
                    self._owns_async_client = True
                except ImportError:
                    semantic = _semantic_fallback("未安装 openai SDK；请执行 python -m pip install openai")
                except ValueError:
                    semantic = _semantic_fallback("API 客户端配置无效或未设置 OPENAI_API_KEY")
                except Exception as exc:
                    semantic = _semantic_fallback(f"LLM 客户端初始化失败（{type(exc).__name__}）")
            if self.async_client is not None:
                options = {k: v for k, v in self.llm_options.items() if k != "client"}
                semantic = await arun_llm_semantic_qa(cleaned, static_errors, client=self.async_client,
                                                     total_timeout=self.total_timeout, **options)
        return self._assemble(caption, sections, cleaned, static_errors, semantic)

    async def aclose(self) -> None:
        """只关闭本 Agent 自建客户端；外部注入客户端由调用者管理。"""
        if self._owns_async_client and self.async_client is not None:
            client, self.async_client = self.async_client, None
            self._owns_async_client = False
            await client.close()

    async def __aenter__(self) -> CaptionQAAgent:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    def _assemble(self, caption: str, sections: Mapping[str, str], cleaned: Mapping[str, str],
                  static_errors: list[Issue], semantic: SemanticResult) -> QAReport:
        coverage = subject_coverage(sections)
        merged = OrderedDict(cleaned)
        guards: list[Issue] = []
        for name, candidate in semantic["corrected_sections"].items():
            if name not in cleaned or candidate.strip() == cleaned[name].strip():
                continue
            old = cleaned[name].strip()
            new = candidate.strip()
            related = [i for i in semantic["issues"] if i["section"] == name]
            reason = ""
            if name not in CORE_SECTIONS:
                reason = "不允许改写非核心字段"
            elif not new or re.search(r"^##\s+", new, re.MULTILINE):
                reason = "订正稿为空或包含额外二级标题"
            elif any(not i["can_auto_fix"] or i["issue_type"] not in AUTO_LLM_ISSUE_TYPES | LEGACY_AUTO_ISSUE_TYPES for i in related):
                reason = "字段中存在自动修复白名单之外或不可自动修复的问题"
            elif name == "主体" and any(i["issue_type"] == "duplicate_defined_subjects" for i in static_errors) and SUBJECT_RE.findall(old) != SUBJECT_RE.findall(new):
                reason = "主体编号重复定义，不能自动删除或重排主体标签"
            elif TIMESTAMP_RE.findall(old) != TIMESTAMP_RE.findall(new):
                reason = "订正稿改变了豁免的时间戳"
            elif re.findall(r"\d+(?:[.,]\d+)?", old) != re.findall(r"\d+(?:[.,]\d+)?", new):
                reason = "订正稿改变了数值结构，不能作为标点硬伤自动修复"
            elif TAG_RE.findall(old) != TAG_RE.findall(new):
                reason = "订正稿改变了已有实体标签的数量、编号或顺序；标签格式仅由静态规则修复"
            elif not _changes_supported(old, new, related):
                reason = "订正稿有超出问题片段的修改，拒绝自动应用"
            elif not _whitelisted_changes(old, new, related):
                reason = "订正稿不符合三类自动修复白名单：仅允许已有裸标签补尖括号、词表错字替换或纯标点修复"
            elif any(i["issue_type"] in {"bare_tags", "duplicate_prefixes", "missing_punctuation"}
                     for i in run_static_rules({name: new})):
                reason = "订正稿重新引入了已被静态修复的格式问题"
            if reason:
                guards.append(_issue("merge", name, old, "unsafe_llm_edit", reason))
            else:
                merged[name] = _pad_like(cleaned[name], candidate)
        # 静态修复在前、模型差异字段覆盖在后；最后只复检，不改动元数据。
        remaining = run_static_rules(merged)
        unresolved_keys = {(i["section"], i["issue_type"], i["original_snippet"]) for i in remaining}
        for item in static_errors:
            item["resolved"] = item["issue_type"] not in HARD_TYPES and (
                item["section"], item["issue_type"], item["original_snippet"]) not in unresolved_keys
        llm_issues: list[Issue] = []
        for item in semantic["issues"]:
            auto = item["can_auto_fix"] and item["issue_type"] in AUTO_LLM_ISSUE_TYPES | LEGACY_AUTO_ISSUE_TYPES
            normalized = _issue("llm", item["section"], item["original_snippet"],
                                item["issue_type"], item["suggestion"], auto)
            if 'replacement' in item:
                normalized['replacement']=item['replacement']
            normalized["resolved"] = auto and _snippet_was_edited(
                cleaned[item["section"]].strip(), merged.get(item["section"], "").strip(), item["original_snippet"])
            llm_issues.append(normalized)
        issues = static_errors + llm_issues + guards
        reasons = [i["suggestion"] for i in issues if not i["resolved"]]
        reasons.extend(i["suggestion"] for i in remaining)
        if semantic["need_human_review"]:
            reasons.append(semantic["human_review_reason"] or "LLM 标记需要人工复核")
        revised = rebuild_caption(caption, merged)
        review = bool(reasons)
        status = "NEED_HUMAN_REVIEW" if review else "AUTO_FIXED" if issues or revised != caption else "PASS"
        return QAReport(caption_parser_version=CAPTION_PARSER_VERSION,status=status, qa_scope="static_only" if self.dry_run else "static_and_semantic",
            field_resolution_notes=list(getattr(sections,'field_resolution_notes',())),
            semantic_qa_performed=semantic["performed"], need_human_review=review,
            human_review_reason="；".join(dict.fromkeys(reasons)), subject_check=coverage,
            static_errors=static_errors, llm_issues=semantic["issues"], issues=issues,
            remaining_static_errors=remaining, revised_caption=revised)

    @staticmethod
    def failure_report(caption: str, reason: str) -> QAReport:
        error = _issue("input", "", "", "input_error", reason)
        return QAReport(caption_parser_version=CAPTION_PARSER_VERSION,status="NEED_HUMAN_REVIEW", qa_scope="input_error", semantic_qa_performed=False,
            need_human_review=True, human_review_reason=reason, subject_check={}, static_errors=[error],
            llm_issues=[], issues=[error], remaining_static_errors=[error], revised_caption=caption)

    def check_batch(self, captions: Sequence[str]) -> list[QAReport]:
        return [self.check(caption) for caption in captions]


def print_markdown_report(report: QAReport, *, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    def emit(text: str) -> None:
        print(text, file=stream)
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>").replace("<", "&lt;").replace(">", "&gt;")
    emit("## 1. 主体与标签核查结果\n")
    emit(f"状态：**{report['status']}**\n")
    emit("质检范围：" + ("仅本地静态预检，未执行语义质检。" if report["qa_scope"] == "static_only"
                         else "静态与语义质检。" if report["semantic_qa_performed"] else "语义质检未完成，需人工复核。") + "\n")
    for key, label in (("defined_subjects", "已定义主体"), ("instruction_referenced_subjects", "自然指令引用主体"),
                       ("missing_subjects", "自然指令漏写已定义主体（重点复核）"), ("ghost_subjects", "幽灵主体")):
        emit(f"- {label}：{'、'.join(report['subject_check'].get(key, [])) or '无'}")
    if report["human_review_reason"]:
        emit("\n人工复核原因：" + report["human_review_reason"])
    emit("\n## 2. 问题清单表格\n")
    emit("| 来源 | 字段 | 类型 | 原文片段 | 建议 | 已修复 |\n| --- | --- | --- | --- | --- | --- |")
    for item in report["issues"]:
        emit("| " + " | ".join(cell(item[k]) for k in
            ("source", "section", "issue_type", "original_snippet", "suggestion", "resolved")) + " |")
    if not report["issues"]:
        emit("| — | — | — | — | 当前检查范围未发现问题 | — |")
    emit("\n## 3. 原文（识别阶段未修改）\n" if report.get('workflow_version') == 'review-first-v1' else "\n## 3. 订正后完整 Caption\n")
    # 动态 fence 防止输入包含反引号导致报告截断。
    longest = max((len(m.group()) for m in re.finditer(r"`+", report["revised_caption"])), default=0)
    fence = "`" * max(3, longest + 1)
    emit(fence + "markdown\n" + report["revised_caption"].rstrip("\r\n") + "\n" + fence)


def _indexed_input_records(path: Path, *, sheet_name: str | None = None) -> Iterator[tuple[int, Any, str | None]]:
    """统一表格读取，JSON 数组兼容旧入口；索引保持原表位置。"""
    if path.suffix.lower() in {".xlsx", ".csv", ".jsonl", ".json"}:
        from captionqa.caption_qa_table_io import iter_table_records
        yield from iter_table_records(path, sheet_name=sheet_name)
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".json":
            data = json.load(stream)
            for number, record in enumerate(data if isinstance(data, list) else [data], 1):
                yield number, record, None
        else:
            raise ValueError("批量输入支持 .xlsx / .csv / .jsonl / .json")


def _input_records(path: Path) -> Iterator[tuple[Any, str | None]]:
    """保留旧同步 CSV 导出入口使用的二元组接口。"""
    for _, record, error in _indexed_input_records(path):
        yield record, error


def _record_dict(record: Any, caption_field: str) -> dict[str, Any]:
    return (dict(record) if isinstance(record, dict) else
            {caption_field: record} if isinstance(record, str) else {"raw_input": record})


def _output_record(record: Any, error: str | None, agent: CaptionQAAgent,
                   caption_field: str, *, report: QAReport | None = None,
                   row_index: int | None = None,
                   checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """只负责结果组装；异步路径传入 report，绝不回退到同步 LLM 调用。"""
    row = _record_dict(record, caption_field)
    if report is None:
        caption = row.get(caption_field)
        if error or not isinstance(caption, str):
            report = agent.failure_report(caption if isinstance(caption, str) else "", error or f"缺少字符串字段 {caption_field}")
        else:
            report = agent.check(caption)
    generated: dict[str, Any] = {"qa_status": report["status"], "qa_need_human_review": report["need_human_review"],
        "qa_human_review_reason": report["human_review_reason"],
        "qa_semantic_qa_performed": report["semantic_qa_performed"], "qa_scope": report["qa_scope"],
        "qa_revised_caption": report["revised_caption"], "qa_report": report}
    generated.update(_qa_result_values(report))
    if row_index is not None:
        generated["row_index"] = row_index
    if checkpoint is not None:
        generated["qa_checkpoint"] = dict(checkpoint)
    # 所有被占用的原列均备份，包括 row_index 和 qa_checkpoint。
    collisions = {k: row[k] for k in generated if k in row}
    if collisions:
        backup = "qa_previous_values"
        while backup in row or backup in generated:
            backup += "_"
        row[backup] = collisions
    row.update(generated)
    return row


def _checkpoint_key(record: Any, row_index: int) -> str:
    """类型进入 key，避免字符串 '1'、整数 1 和后备行号 1 相互串台。"""
    identifier = record.get("id") if isinstance(record, dict) else None
    if identifier is None or identifier == "":
        identity: list[Any] = ["row_index", row_index]
    elif type(identifier) in (str, int):
        identity = ["id", type(identifier).__name__, identifier]
    else:
        raise ValueError(f"第 {row_index} 条 id 必须是字符串或整数；空值可使用后备行号")
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_context(source: Path, agent: CaptionQAAgent, caption_field: str, *,
                        sheet_name: str | None = None) -> dict[str, Any]:
    scope = "static_only" if agent.dry_run else "static_and_semantic"
    options = agent.llm_options
    # 仅保存配置摘要，不把 API key 或可能含私密信息的 base_url 写入输出。
    identity = {} if agent.dry_run else {
        "model": options.get("model_name") or os.getenv("MODEL_NAME") or "gpt-4o-mini",
        "base_url": options.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    }
    return {"version": 1, "auto_fix_policy": AUTO_FIX_POLICY_VERSION,
            "source_sha256": _source_sha256(source), "caption_field": caption_field,
            "qa_scope": scope, "human_caption_filter": CAPTION_SCOPE_VERSION, "sheet_name": sheet_name,
            "model_identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest(),
            **({"workflow_version": agent.workflow_version} if hasattr(agent, "workflow_version") else {})}


def _load_completed_checkpoints(path: Path, context: Mapping[str, Any]) -> tuple[set[str], bool]:
    """校验断点来源；只修复末尾撕裂行，中间损坏或来源不符则拒绝续写。

    返回已完成 key 和追加前是否需要补换行。检查完整文件后才截断损坏尾行。
    """
    completed: set[str] = set()
    if not path.exists():
        return completed, False
    truncate_at: int | None = None
    append_newline = False
    with path.open("rb") as stream:
        number = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            number += 1
            append_newline = not line.endswith(b"\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8-sig" if start == 0 else "utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if any(remainder.strip() for remainder in stream):
                    raise ValueError(f"断点文件第 {number} 行损坏，不能跳过中间记录") from None
                truncate_at = start
                append_newline = False
                break
            metadata = row.get("qa_checkpoint") if isinstance(row, dict) else None
            if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in context.items()):
                raise ValueError(f"断点文件第 {number} 行缺少断点信息，或来源/配置不匹配，请使用新的输出文件")
            if not isinstance(metadata.get("key"), str) or type(metadata.get("completed")) is not bool:
                raise ValueError(f"断点文件第 {number} 行缺少合法 key/completed 标识")
            index = row.get("row_index")
            if type(index) is not int or index < 1 or _checkpoint_key(row, index) != metadata["key"]:
                raise ValueError(f"断点文件第 {number} 行 key 与 id/row_index 不一致")
            if metadata["completed"]:
                if instruction_parser_outdated(row.get(metadata['caption_field']),
                        row.get('qa_report',{}).get('caption_parser_version',1),
                        semantic=row.get('qa_report',{}).get('semantic_qa_performed',False)):
                    raise ValueError(f"断点文件第 {number} 行使用旧版字段归属解析，请使用新的输出文件重新检测；旧文件保留")
                completed.add(metadata["key"])
    if truncate_at is not None:
        with path.open("r+b") as stream:
            stream.truncate(truncate_at)
    return completed, append_newline


async def aprocess_batch(input_path: str | Path, output_path: str | Path, agent: CaptionQAAgent,
                         *, caption_field: str | None = None, column: str | None = None,
                         concurrency: int = 10, sheet_name: str | None = None,
                         checkpoint_path: str | Path | None = None, id_column: str | None = None,
                         on_progress: Any = None) -> int:
    """并发检查单条 Caption，完成一条立即追加并 flush；返回本次新写入条数。

    JSONL 为断点事实来源，输出按完成顺序；原始顺序可通过 row_index 还原。
    语义请求失败写入 completed=false，下次自动重试；确定性输入错误标记完成。
    相同输出文件同一时刻应只运行一个批处理进程。CSV 导出使用 process_batch。
    """
    tasks: list[asyncio.Task[Any]] = []
    try:
        source, target = Path(input_path), Path(output_path)
        if target.suffix.lower() == ".txt":
            from captionqa.caption_qa_id_list import detect_id_column, export_id_list
            identifier_column = detect_id_column(source, sheet_name=sheet_name, column=id_column)
            cache = Path(checkpoint_path) if checkpoint_path is not None else target.with_suffix(".jsonl")
            if cache.suffix.lower() != ".jsonl" or cache.resolve() in {source.resolve(), target.resolve()}:
                raise ValueError("ID 名单的续传缓存必须是独立 JSONL 文件")
            if target.resolve() == source.resolve():
                raise ValueError("输入和输出不能是同一文件")
            total = sum(1 for _ in _indexed_input_records(source, sheet_name=sheet_name))
            count = await aprocess_batch(source, cache, agent, caption_field=caption_field, column=column,
                                        concurrency=concurrency, sheet_name=sheet_name,
                                        id_column=identifier_column, on_progress=on_progress)
            export_id_list(cache, target, id_column=identifier_column, total=total,
                           dry_run=agent.dry_run, sheet_name=sheet_name)
            return count
        if target.suffix.lower() in {".xlsx", ".csv"}:
            return await _aprocess_table(source, target, agent, caption_field=caption_field, column=column,
                concurrency=concurrency, sheet_name=sheet_name, checkpoint_path=checkpoint_path,
                id_column=id_column, on_progress=on_progress)
        if checkpoint_path is not None and Path(checkpoint_path).resolve() != target.resolve():
            raise ValueError("JSONL 输出本身就是续传缓存，不应另指定不同的 checkpoint_path")
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency 必须是正整数")
        if column is not None and caption_field is not None and column != caption_field:
            raise ValueError("column 与 caption_field 不能指定不同列")
        manual_column = column if column is not None else caption_field
        if manual_column == "":
            raise ValueError("column 不能为空")
        if manual_column is not None and _excluded_caption_column(manual_column):
            raise ValueError("指定列属于备注、机标或 QA 结果列；请明确指定 Caption 列。未调用 API。")
        if source.resolve() == target.resolve():
            raise ValueError("输入和输出不能是同一文件")
        if source.suffix.lower() not in (".xlsx", ".json", ".jsonl", ".csv") or target.suffix.lower() != ".jsonl":
            raise ValueError("异步断点批处理输入支持 .xlsx/.json/.jsonl/.csv，缓存输出必须为 .jsonl")
        selected_column = manual_column or detect_caption_column(source, sheet_name=sheet_name)
        context = _checkpoint_context(source, agent, selected_column, sheet_name=sheet_name)
        # 流式预检重复 id，防止同一轮产生重复请求或在恢复时静默漏掉重号记录。
        seen: set[str] = set()
        columns_found: set[str] = set()
        for row_index, record, _ in _indexed_input_records(source, sheet_name=sheet_name):
            columns_found.update(_record_dict(record, selected_column))
            key = _checkpoint_key(record, row_index)
            if key in seen:
                raise ValueError(f"第 {row_index} 条 id 重复；请修正输入后重试")
            seen.add(key)
        if manual_column is not None and manual_column not in columns_found:
            raise ValueError(f'指定的 Caption 列不存在："{manual_column}"；请检查 --column 或使用自动检测')
        if _source_sha256(source) != context["source_sha256"]:
            raise ValueError("输入文件在预检期间发生变更，请停止编辑后重试")
        del seen
        completed, append_newline = _load_completed_checkpoints(target, context)
        target.parent.mkdir(parents=True, exist_ok=True)
        queue: asyncio.Queue[tuple[int, Any, str | None] | None] = asyncio.Queue(maxsize=concurrency * 2)
        semaphore, write_lock = asyncio.Semaphore(concurrency), asyncio.Lock()
        count = 0

        async def produce() -> None:
            for item in _indexed_input_records(source, sheet_name=sheet_name):
                if _checkpoint_key(item[1], item[0]) not in completed:
                    await queue.put(item)
            for _ in range(concurrency):
                await queue.put(None)

        with target.open("a", encoding="utf-8", newline="") as out:
            if append_newline:
                out.write("\n")
                out.flush()

            async def worker() -> None:
                nonlocal count
                while True:
                    item = await queue.get()
                    try:
                        if item is None:
                            return
                        row_index, record, error = item
                        request_stats = {}
                        used_column, caption = _select_human_caption(record, selected_column)
                        if error or used_column is None:
                            report = _skipped_report()
                            is_completed = True
                        else:
                            started = time.monotonic()
                            def progress(event):
                                if event['state'] == 'timed_out':
                                    request_stats['timed_out'] = True
                                if 'attempt' in event:
                                    request_stats['attempts'] = event['attempt']
                                request_stats['elapsed_seconds'] = round(time.monotonic()-started, 2)
                                if on_progress is not None:
                                    identifier = record.get(id_column) if isinstance(record, Mapping) and id_column else None
                                    if identifier is None and isinstance(record, Mapping):
                                        identifier = next((record[k] for k in ('id','样本ID','题目ID','sample_id')
                                                           if record.get(k) is not None), None)
                                    try:
                                        on_progress(dict(event, row_index=row_index, sample_id=identifier,
                                                         elapsed_seconds=time.monotonic()-started))
                                    except Exception:
                                        pass
                            try:
                                async with semaphore:
                                    started = time.monotonic()
                                    token = _REQUEST_PROGRESS.set(progress)
                                    progress({'state':'checking'})
                                    try:
                                        report = await agent.acheck_single_caption(caption)
                                    finally:
                                        _REQUEST_PROGRESS.reset(token)
                                is_completed = (agent.dry_run or report["semantic_qa_performed"] or
                                                report["qa_scope"] == "input_error")
                            except Exception as exc:
                                # 不回显异常正文，避免第三方错误信息夹带请求或密钥。
                                report = agent.failure_report(caption, f"异步质检未完成（{type(exc).__name__}），下次续传将重试")
                                is_completed = False
                            finally:
                                progress({'state':'finished'})
                        checkpoint = dict(context, key=_checkpoint_key(record, row_index), completed=bool(is_completed),
                                          selected_caption_column=used_column)
                        if request_stats:
                            checkpoint['request_stats'] = request_stats
                        row = _output_record(record, error, agent, selected_column, report=report,
                                             row_index=row_index, checkpoint=checkpoint)
                        if used_column is None or error:
                            cell = _record_dict(record, selected_column).get(selected_column)
                            reason = ("input_record_error" if error else
                                      "missing_caption_column" if selected_column not in _record_dict(record, selected_column) else
                                      "empty_caption" if cell is None or (isinstance(cell, str) and not cell.strip()) else
                                      "table_header" if isinstance(cell, str) and cell.strip().lower() in {"caption", "人标caption", "人标 caption"} else
                                      "nontext_caption" if not isinstance(cell, str) else "unsupported_caption_format")
                            row["qa_skip_reason"] = reason
                        encoded = json.dumps(row, ensure_ascii=False, default=str) + "\n"
                        async with write_lock:
                            out.write(encoded)
                            out.flush()
                            count += 1
                    finally:
                        queue.task_done()

            tasks = [asyncio.create_task(produce())] + [asyncio.create_task(worker()) for _ in range(concurrency)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        return count
    finally:
        await agent.aclose()


async def _aprocess_table(input_path: Path, output_path: Path, agent: CaptionQAAgent, *,
                          caption_field: str | None = None, column: str | None = None,
                          concurrency: int = 10, sheet_name: str | None = None,
                          checkpoint_path: str | Path | None = None, id_column: str | None = None,
                          on_progress: Any = None) -> int:
    """JSONL 是逐条落盘的事实来源，表格仅在缓存处理完成后原位追加结果列。"""
    from captionqa.caption_qa_table_io import export_qa_table
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出不能是同一文件")
    cache = Path(checkpoint_path) if checkpoint_path is not None else output_path.with_suffix(".jsonl")
    if cache.suffix.lower() != ".jsonl" or cache.resolve() in {input_path.resolve(), output_path.resolve()}:
        raise ValueError("续传缓存必须是独立 .jsonl 文件，不能覆盖输入或最终结果表")
    if output_path.suffix.lower() == ".xlsx":
        try:
            import openpyxl  # noqa: F401 先确认依赖存在，避免调用 API 后才发现无法导出。
        except ImportError as exc:
            raise ValueError("Excel 支持需要 openpyxl，请运行 python -m pip install -r requirements.txt") from exc
    count = await aprocess_batch(input_path, cache, agent, caption_field=caption_field, column=column,
                                concurrency=concurrency, sheet_name=sheet_name,
                                id_column=id_column, on_progress=on_progress)
    results: dict[int, Mapping[str, Any]] = {}
    with cache.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                results[row["row_index"]] = _qa_result_values(row["qa_report"])
    export_qa_table(input_path, output_path, results, sheet_name=sheet_name)
    print(f"[Checkpoint] 续传缓存: {cache}")
    return count


def process_batch(input_path: str | Path, output_path: str | Path, agent: CaptionQAAgent,
                  *, caption_field: str | None = None, column: str | None = None,
                  concurrency: int = 10, sheet_name: str | None = None,
                  checkpoint_path: str | Path | None = None, id_column: str | None = None) -> int:
    """同步入口；所有表格输出均复用异步并发、前置过滤和 JSONL 缓存。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aprocess_batch(input_path, output_path, agent, caption_field=caption_field,
            column=column, concurrency=concurrency, sheet_name=sheet_name, checkpoint_path=checkpoint_path,
            id_column=id_column))
    raise ValueError("当前已有 asyncio 事件循环，请改用 await aprocess_batch(...)")


def main(demo_caption: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="仅本地规则检查和安全格式修正，不导入或调用 OpenAI")
    parser.add_argument("--review-all", action="store_true", help="所有修改均交人工确认，关闭默认白名单自动修正")
    parser.add_argument("--legacy-auto-fix", action="store_true", help="兼容旧版候选修正接口；默认使用当前安全白名单自动修正")
    parser.add_argument("--text", help="直接传入单条 Markdown Caption")
    parser.add_argument("--caption-file", type=Path, help="单条 UTF-8 Markdown/txt 文件")
    parser.add_argument("--input", type=Path, help="完整业务表 .xlsx/.csv/.jsonl（兼容 .json）")
    parser.add_argument("--output", type=Path, help="批量默认输出问题 ID 名单 .txt；兼容显式 .xlsx/.csv/.jsonl；单条 .json/.md")
    parser.add_argument("--id-column", default=None, help="样本 ID 列名，Excel 也可填列字母；默认自动识别")
    parser.add_argument("--column", "--caption-field", dest="column", default=None,
                        help="手动指定人标列；省略时自动嗅探前 50 条非空数据")
    parser.add_argument("--sheet", default=None, help="Excel 工作表名，默认第一个工作表")
    parser.add_argument("--checkpoint", type=Path, default=None, help="可选：指定 JSONL 续传缓存路径")
    parser.add_argument("--model", default=None, help="覆盖 MODEL_NAME，默认 gpt-4o-mini")
    parser.add_argument("--base-url", default=None, help="覆盖 OPENAI_BASE_URL")
    parser.add_argument("--max-retries", type=int, default=3, help="额外重试次数，范围 0-3，默认 3")
    parser.add_argument("--concurrency", type=int, default=10, help="JSONL 批量最大并发数，默认 10")
    parser.add_argument("--timeout", type=float, default=60.0, help="每次调用超时秒数，默认 60")
    args = parser.parse_args(argv)
    if sum(value is not None for value in (args.text, args.caption_file, args.input)) > 1:
        parser.error("--text、--caption-file、--input 只能选择一个")
    if not 0 <= args.max_retries <= 3 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.concurrency < 1:
        parser.error("max-retries 必须在 0-3 之间，timeout 必须为有限正数，concurrency 必须为正整数")
    if args.legacy_auto_fix:
        agent_class = CaptionQAAgent
    else:
        from captionqa.caption_qa_review import IdentificationAgent, SafeAutoFixAgent
        agent_class = IdentificationAgent if args.review_all else SafeAutoFixAgent
    agent = agent_class(dry_run=args.dry_run, model_name=args.model,
        base_url=args.base_url, max_retries=args.max_retries, timeout=args.timeout)
    try:
        if args.input:
            if args.output is None:
                args.output = args.input.with_name(args.input.stem + "_问题ID名单.txt")
            count = process_batch(args.input, args.output, agent, column=args.column, concurrency=args.concurrency,
                                  sheet_name=args.sheet, checkpoint_path=args.checkpoint, id_column=args.id_column)
            print(f"本次新增处理 {count} 条，结果保存至 {args.output}")
        else:
            if args.output and args.output.suffix.lower() not in (".json", ".md"):
                raise ValueError("单条报告输出支持 .json 或 .md")
            if args.output and args.caption_file and args.output.resolve() == args.caption_file.resolve():
                raise ValueError("输入和输出不能是同一文件")
            caption = args.text if args.text is not None else args.caption_file.read_bytes().decode("utf-8-sig") if args.caption_file else demo_caption
            report = agent.check(caption)
            print_markdown_report(report)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", encoding="utf-8", newline="") as out:
                    if args.output.suffix.lower() == ".json":
                        json.dump(report, out, ensure_ascii=False, indent=2)
                    else:
                        print_markdown_report(report, stream=out)
    except KeyboardInterrupt:
        print("已中断；JSONL 中已落盘的记录将在下次运行时自动跳过。", file=sys.stderr)
        return 130
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 2
    return 0


# 原样保留用户提供的真实样例，包括原有硬伤与省略号，方便立即对照验证。
DEMO_CAPTION = """## 任务类型 
人物参考|造型参考|动作参考
## 提示
参考<视频1>和<图片1>生成一段白袍染血的人群倒地挣扎并有一人奔跑出逃的视频。
## similarity_label
 yes
## video_similarity_tag 
实体/IP - 相同|场景 - 相同|动作 - 相同|剧情-相同
## audio_similarity_tag 
无声音
## 输入描述 
<图片1>：展示昏暗山洞学堂中多名身穿白色长袍的群演，其中一名群演白色长袍的胸口与下摆处附着高饱和度的红色喷溅血迹。
<视频1>：展示昏暗山洞学堂中多名身穿白色立领长袍的演员在台阶与地面上依次做出、痛苦挣扎与慌乱奔跑的表演动作，背景可见写有标语的黑板、点燃的烛台。顶部悬挂着两盏大小不一的白色布蔓灯。视频全程无声
## 基调
写实风格的暗调电影感短片，冷暗环境中白袍、红血与暖烛光形成强烈冲突
## 场景
<场景1>：<场景1>：低亮度冷暗色调的昏暗山洞学堂。群演站在石质台阶与泥土地面上...
## 主体
<主体1>：为<视频1>中脱离人群向前奔跑的年轻东亚女性，乌黑长发束成马尾垂在脑后，圆润的脸庞上眉头紧锁、身穿白色立领盘扣中式长袍，袍身宽松垂坠长及脚踝，脚穿白色布鞋。
## 画面元素
 无
## 总结
一群群演在昏暗学堂中捂胸倒地、痛苦挣扎，<主体1>脱离人群提起裙摆跑向镜头。
## 自然指令
0-4秒，全景镜头伴随小幅度下降。身穿白色立领长袍的群演胸口与下摆附着中弹后的暗红色红色喷溅血迹...画面中央偏右的<主体1>面带惊恐从后排人群中站起，提起长袍裙摆迅速跑向画面中央...视频全程无声。
"""

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main(DEMO_CAPTION))
