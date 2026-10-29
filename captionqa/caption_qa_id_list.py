"""Plain-text problem-ID output. Checkpoints stay local; no result table."""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from captionqa.caption_qa_summary import build_check_report
from captionqa.caption_qa_people import build_people_report, export_people_report


def has_content_problem(row):
    report = row.get("qa_report", {})
    if row.get("qa_scope") == "input_error":
        return False
    return (any(issue.get("source") in {"static", "llm"} for issue in report.get("issues", []))
            or bool(report.get("semantic_qa_performed") and report.get("need_human_review")))


def detect_id_column(source, *, sheet_name=None, column=None):
    from captionqa.caption_qa_table_io import iter_table_records, table_columns
    source = Path(source)
    columns = table_columns(source, sheet_name=sheet_name)
    if column:
        if column in columns:
            return column
        if source.suffix.lower() == ".xlsx" and re.fullmatch(r"[A-Za-z]{1,3}", column):
            from openpyxl.utils import column_index_from_string
            index = column_index_from_string(column) - 1
            if index < len(columns):
                return columns[index]
        raise ValueError("指定的 ID 列不存在，请检查 ID 列名或 Excel 列字母。未调用 API。")
    priorities = {"样本id": 3, "样本编号": 3, "sample_id": 3, "sampleid": 3,
                  "id": 2, "任务id": 2, "task_id": 2, "video_id": 1, "视频id": 1}
    ranked = [(priorities.get(re.sub(r"\s+", "", name).lower(), 0), name) for name in columns]
    best = max((score for score, _ in ranked), default=0)
    if best:
        candidates = [name for score, name in ranked if score == best]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("发现多个 ID 列，请在更多设置中明确指定 ID 列。未调用 API。")
    # This business export has a broken '=SUM()' header over its sample IDs.
    # Recognize its strongly typed cm/cmd IDs, never substitute row numbers.
    if columns:
        values = []
        for index, row, _ in iter_table_records(source, sheet_name=sheet_name):
            value = row.get(columns[0])
            if value is not None and str(value).strip():
                values.append(str(value))
            if index >= 50:
                break
        if len(values) >= 2 and all(re.fullmatch(r"cm[d]?[0-9a-f]{6,}", value) for value in values):
            return columns[0]
    raise ValueError("未能确认样本 ID 列，请在更多设置中填写 ID 列名或 Excel 列字母。未调用 API。")


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _pending_content_notes(row):
    """Only unresolved findings belong in the user-facing explanation."""
    report = row.get("qa_report", {})
    labels = {"missing_subjects": "自然指令漏写已定义主体", "ghost_subjects": "引用未定义主体",
              "duplicate_defined_subjects": "主体重号", "skipped_numbering": "主体编号断层",
              "bare_tags": "裸标签缺少 <>", "missing_punctuation": "句末标点缺失",
              "symbol_format": "内容符号不成对", "symbol_width": "全角/半角符号",
              "label_separator": "多标签拼接符不规范",
              "unsafe_llm_edit": "修订未应用"}
    notes = []
    for item in report.get("issues", []):
        if item.get("resolved"):
            continue
        kind = labels.get(item.get("issue_type"), item.get("issue_type", "待复核问题"))
        location = item.get("section", "")
        prefix = f"[{location} / {kind}]" if location else f"[{kind}]"
        # Merge guards may contain an entire statically revised paragraph;
        # explain why review is needed without exporting that revised text.
        snippet = item.get("original_snippet", "") if item.get("source") != "merge" else ""
        note = prefix + (f' 原文片段："{snippet}"；' if snippet else " ") + item.get("suggestion", "请人工核对")
        if note not in notes:
            notes.append(note)
    if not notes and report.get("need_human_review"):
        reason = report.get("human_review_reason") or row.get("qa_human_review_reason")
        if reason:
            notes.append(reason)
    return "；".join(notes)


def export_id_list(cache, output, *, id_column, total, dry_run, sheet_name=None):
    cache, output = Path(cache), Path(output)
    latest = {}
    with cache.open("rb") as stream:
        for line in stream:
            if line.strip() and line.endswith(b"\n"):
                row = json.loads(line)
                latest[row["row_index"]] = row
    resolutions = {}
    resolution_path = cache.with_suffix(".id_resolutions.json")
    if resolution_path.exists():
        document = json.loads(resolution_path.read_text(encoding="utf-8"))
        source_hashes = {row.get("qa_checkpoint", {}).get("source_sha256") for row in latest.values()}
        if source_hashes != {document.get("source_sha256")}:
            raise ValueError("ID 核对记录与输入文件不匹配，拒绝应用。")
        resolutions = document.get("rows", {})
    ids, details, skipped, skipped_rows, unfinished, missing, resolved = [], [], Counter(), [], [], [], []
    instruction_subject_alerts = []
    labels = {"empty_caption": "Caption 为空", "unsupported_caption_format": "不符合支持的分段 Caption 格式",
              "missing_caption_column": "缺少 Caption 列", "nontext_caption": "Caption 不是文本",
              "input_record_error": "原始记录无法解析", "table_header": "多级表头行"}
    for index, row in sorted(latest.items()):
        identifier = row.get(id_column)
        usable_id = type(identifier) in (str, int) and str(identifier).strip() and not re.search(r"[\r\n]", str(identifier))
        if not usable_id and str(index) in resolutions:
            resolution = resolutions[str(index)]
            identifier = resolution.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or re.search(r"[\r\n]", identifier):
                raise ValueError("ID 核对记录缺少合法的样本 ID。")
            usable_id = True
            resolved.append({"excel_row": index + 1, **resolution})
        label = str(identifier) if usable_id else f"未提供 ID（原表第 {index + 1} 行）"
        if row.get("qa_status") == "SKIPPED_NOT_HUMAN_CAPTION":
            reason = labels.get(row.get("qa_skip_reason"), "不符合支持的 Caption 格式")
            skipped[reason] += 1
            skipped_rows.append(f"{label}：{reason}")
            continue
        report = row.get("qa_report", {})
        missing_subjects = report.get("subject_check", {}).get("missing_in_instruction", [])
        if missing_subjects:
            instruction_subject_alerts.append({"id": str(identifier) if usable_id else None,
                "excel_row": index + 1, "missing_subjects": missing_subjects})
        if row.get("qa_scope") == "input_error" or not row.get("qa_checkpoint", {}).get("completed"):
            unfinished.append(f"{label}：{row.get('qa_human_review_reason') or '检测未完成'}")
        if has_content_problem(row):
            if not usable_id:
                missing.append(index + 1)
            elif str(identifier) not in ids:
                ids.append(str(identifier))
            reasons = _pending_content_notes(row)
            if reasons:
                details.append(f"{label}（原表第 {index + 1} 行）：{reasons}")
    summary = {"problem_ids": len(ids), "processed": len(latest), "total": total,
               "skipped_reasons": dict(skipped), "not_checked": len(unfinished),
               "missing_problem_ids": missing, "dry_run": dry_run,
               "complete": len(latest) == total and not unfinished and not missing,
               "resolved_ids": resolved,
               "instruction_subject_alerts": instruction_subject_alerts,
               "check_report": build_check_report(latest.values())}
    mode = "确定性底线检查，不执行模型语义检查" if dry_run else "辅助筛查（含模型疑似问题，不作底线结论）"
    lines = ["Caption 问题 ID 名单说明", f"工作表：{sheet_name or '默认'}", f"范围：仅指定 Caption 列；{mode}",
             f"已处理：{len(latest)} / {total} 行；问题 ID：{len(ids)} 个",
             "名单包括已修复和需人工复核的内容问题；跳过及纯检测异常不计为内容问题。",
             "本说明仅逐条列出待复核问题；已完成的修复只保留数量统计，不列原文、修改建议或修订内容。",
             "跳过不等于通过。ID 文件为空仅表示已检测部分未发现内容问题，须同时查看检测覆盖范围。"]
    if not summary["complete"]:
        lines.append("注意：存在未完成检测或缺失 ID，本名单不是完整结论。")
    if dry_run:
        lines.append("底线只检查约定的标签格式、句末标点、内容符号配对和主体定义/引用规则，不评价病句、疑似错字或画面事实。跳过或解析失败的记录未完成验收。")
    lines += ["", "各检查项简报（按 Caption 计数，同条可含多项，各项不可直接相加）："]
    for item in summary["check_report"]["items"]:
        if dry_run and item["requires_semantic"]:
            continue
        if item["requires_semantic"] and not summary["check_report"]["semantic_checked"]:
            lines.append(f"{item['label']}：未执行语义检查。")
        else:
            lines.append(f"{item['label']}：涉及 {item['affected']} 条，已修复 {item['fixed']} 条，需复核 {item['review']} 条。")
    for key, label in (("correction_review_rows", "修订未应用，需人工确认"), ("other_review_rows", "其他人工复核说明")):
        if summary["check_report"][key]:
            lines.append(f"{label}：{summary['check_report'][key]} 条。")
    if instruction_subject_alerts:
        lines += ["", "【重点复核：自然指令漏写已定义主体】",
                  "以下主体已在“主体”中定义，但未在“自然指令”中引用；总结中的引用不能替代自然指令。",
                  "仅提示人工核对，不自动补标签或编造动作。"]
        for alert in instruction_subject_alerts:
            label = alert["id"] if alert["id"] is not None else "未提供 ID"
            lines.append(f"ID {label}（原表第 {alert['excel_row']} 行）：自然指令缺少 "
                         + "、".join(alert["missing_subjects"]) + "。")
    lines += ["", "跳过原因："] + [f"{reason}：{count} 行" for reason, count in skipped.items()]
    lines += ["跳过行（未检测，不计为内容问题）："] + (skipped_rows or ["无"])
    lines += ["", "未完成检测（不据此判定内容有问题）："] + (unfinished or ["无"])
    if missing:
        lines += ["缺少 ID 的问题行：" + "、".join(map(str, missing))]
    if resolved:
        lines += ["", "ID 核对来源（仅补充输出名单，未修改原文件）："]
        for item in resolved:
            lines.append(f"原表第 {item['excel_row']} 行 → {item['id']}；{item.get('provenance', '已核对的 ID 来源')}。")
    lines += ["", "待复核问题说明："] + (details or ["当前已检测范围内无待复核的内容问题；已完成的修复不逐条列出。"])
    _write_atomic(output, "".join(identifier + "\n" for identifier in ids))
    explanation = output.with_name(output.stem + "_检测说明.txt")
    _write_atomic(explanation, "\n".join(lines) + "\n")
    summary["explanation_path"] = str(explanation)
    people_report = build_people_report(latest.values())
    summary["people_report_path"] = export_people_report(people_report, output)
    _write_atomic(cache.with_suffix(".summary.json"), json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
