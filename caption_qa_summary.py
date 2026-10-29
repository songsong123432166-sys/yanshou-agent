"""Per-check Caption counts from saved reports; never calls a model."""
from collections import Counter


CHECKS = (
    ("punctuation", "标点问题", False),
    ("bare_tags", "标签格式（尖括号 / 空格）", False),
    ("symbols", "内容符号规范", False),
    ("typos", "疑似错别字", True),
    ("grammar", "病句 / 语句不通顺", True),
    ("missing_subjects", "自然指令漏写已定义主体", False),
    ("implicit_subjects", "隐式漏挂主体标签", True),
    ("duplicate_subjects", "主体重号", False),
    ("ghost_subjects", "引用未定义主体", False),
    ("numbering", "主体编号断层", False),
)
ISSUE_CHECK = {
    "missing_punctuation": "punctuation", "标点符号硬伤": "punctuation",
    "bare_tags": "bare_tags", "裸标签漏写 <>": "bare_tags",
    "symbol_format": "symbols", "symbol_width": "symbols", "label_separator": "symbols", "内容符号不规范": "semantic_symbols",
    "无歧义错别字": "typos", "语法病句与语句不通顺": "grammar",
    "missing_subjects": "missing_subjects", "隐式漏挂主体标签": "implicit_subjects",
    "duplicate_defined_subjects": "duplicate_subjects", "ghost_subjects": "ghost_subjects",
    "skipped_numbering": "numbering",
    # Historical combined types cannot be accurately split into current checks.
    "错别字与衍字": "legacy", "标点与标签格式": "legacy",
}
DISABLED = {"duplicate_prefixes", "重复标签前缀", "衍字/叠词冗余", "跨字段属性冲突", "属性冲突"}


def row_check_state(row):
    report = row.get("qa_report", {})
    if (not report or row.get("qa_status") == "SKIPPED_NOT_HUMAN_CAPTION"
            or row.get("qa_scope") == "input_error" or report.get("qa_scope") == "input_error"):
        return None
    checks = {}
    correction_review = False
    for issue in report.get("issues", []):
        kind = issue.get("issue_type")
        if issue.get("source") == "merge":
            correction_review |= not issue.get("resolved", False)
            continue
        if issue.get("source") not in {"static", "llm"} or kind in DISABLED:
            continue
        key = ISSUE_CHECK.get(kind, "other")
        # One unresolved occurrence makes this check require review for this row.
        checks[key] = checks.get(key, False) or not issue.get("resolved", False)
    semantic = bool(report.get("semantic_qa_performed"))
    other_review = (semantic and report.get("need_human_review", False)
                    and not any(checks.values()) and not correction_review
                    and not any(i.get("issue_type") in DISABLED for i in report.get("issues", [])))
    return {"checks": checks, "semantic": semantic, "correction_review": correction_review,
            "other_review": other_review}


def aggregate_check_states(states):
    states = [state for state in states if state is not None]
    affected, review = Counter(), Counter()
    for state in states:
        for key, unresolved in state["checks"].items():
            affected[key] += 1
            review[key] += bool(unresolved)
    definitions = list(CHECKS)
    for key, label in (("legacy", "旧版混合类型"), ("other", "其他内容问题"),
                       ("semantic_symbols", "疑似内容符号用错")):
        if affected[key]:
            definitions.append((key, label, True))
    return {"version": 1, "unit": "caption_rows", "checked": len(states),
            "semantic_checked": sum(state["semantic"] for state in states),
            "items": [{"key": key, "label": label, "affected": affected[key],
                       "fixed": affected[key] - review[key], "review": review[key],
                       "requires_semantic": requires_semantic}
                      for key, label, requires_semantic in definitions],
            "correction_review_rows": sum(state["correction_review"] for state in states),
            "other_review_rows": sum(state["other_review"] for state in states)}


def build_check_report(rows):
    # Checkpoint retries append new versions; only the latest row is authoritative.
    latest = {row["row_index"]: row for row in rows}
    return aggregate_check_states(row_check_state(row) for row in latest.values())
