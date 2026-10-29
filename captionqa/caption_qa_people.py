"""Local annotator QA counts, with explicit attribution and denominators."""
import re
from collections import Counter
from datetime import date
from fractions import Fraction

from captionqa.caption_qa_summary import aggregate_check_states, row_check_state


def _column_priority(value):
    if not isinstance(value, str):
        return 0
    label = re.sub(r"[\s_]+", "", re.sub(r"__\d+$", "", value)).lower()
    if any(word in label for word in ("音频", "质检", "审核", "复核", "audio", "review")):
        return 0
    if label in {"caption标注员", "caption标注人", "人标标注员", "人工标注员", "captionannotator"}:
        return 2
    return 1 if label in {"标注员", "标注人", "标注人员", "annotator"} else 0


def _person(value):
    if not isinstance(value, str) or not value.strip():
        return None, "未填写标注员"
    value = value.strip()
    if len(value) > 100:
        return None, "人员字段格式异常"
    # Do not choose the last name or charge all co-authors for the same record.
    if any(char in value for char in "\n\r、,，;；|&＆"):
        return None, "多人记录，归属待确认"
    value = value.rstrip("+＋").strip()
    suffix = re.search(r"\s*[(（]?\s*(?:(20\d{2})[./-])?(\d{1,2})[./月-](\d{1,2})日?\s*[)）]?\s*$", value)
    if suffix and value[:suffix.start()].strip():
        try:
            date(int(suffix[1] or 2000), int(suffix[2]), int(suffix[3]))
        except ValueError:
            pass
        else:
            value = value[:suffix.start()].strip()
    else:
        compact = re.search(r"\s*(\d{2})(\d{2})$", value)
        if compact and value[:compact.start()].strip():
            try:
                date(2000, int(compact[1]), int(compact[2]))
            except ValueError:
                pass
            else:
                value = value[:compact.start()].strip()
    value = value.rstrip("+＋").strip()
    if not value:
        return None, "人员字段格式异常"
    if "/" in value:
        return None, "多人或不明确的人员字段"
    return value, None


class PeopleTracker:
    """Keep short metadata and QA flags, never copies of Caption bodies."""

    def __init__(self):
        self.rows = {}

    def add(self, row):
        report = row.get("qa_report", {})
        scope = row.get("qa_scope", report.get("qa_scope"))
        completed = bool(row.get("qa_checkpoint", {}).get("completed"))
        checked = completed and scope != "input_error" and (
            bool(report.get("semantic_qa_performed")) or scope == "static_only")
        problem = (any(i.get("source") in {"static", "llm"} for i in report.get("issues", []))
                   or bool(report.get("semantic_qa_performed") and report.get("need_human_review")))
        caption_keys = {"caption", row.get("qa_checkpoint", {}).get("caption_field"),
                        row.get("qa_checkpoint", {}).get("selected_caption_column")}
        self.rows[row["row_index"]] = {
            "metadata": {k: v for k, v in row.items() if k not in caption_keys
                         and (_column_priority(k) or re.fullmatch(r"__column_\d+(?:__\d+)?", k)
                              or (row.get("qa_skip_reason") == "table_header" and _column_priority(v)))
                         and isinstance(v, str) and len(v) <= 200},
            "header": row.get("qa_skip_reason") == "table_header",
            "eligible": bool(report) and row.get("qa_status") != "SKIPPED_NOT_HUMAN_CAPTION",
            "checked": checked, "problem": checked and problem,
            "review": checked and problem and bool(report.get("need_human_review")),
            "fixed": checked and row.get("qa_status") == "AUTO_FIXED",
            "state": row_check_state(row),
            "baseline": scope == "static_only",
        }

    def snapshot(self):
        candidates = {}
        for row in self.rows.values():
            for key in row["metadata"]:
                if _column_priority(key):
                    candidates[key] = _column_priority(key)
        header = self.rows.get(1)
        if header and header["header"]:
            for key, value in header["metadata"].items():
                if _column_priority(value):
                    candidates[key] = max(candidates.get(key, 0), _column_priority(value))
        if not candidates:
            return {"available": False, "reason": "未找到可确认的标注员列"}
        best = max(candidates.values())
        columns = [key for key, score in candidates.items() if score == best]
        if len(columns) != 1:
            return {"available": False, "reason": "存在多个标注员列，人员归属需确认"}
        column = columns[0]
        groups, unassigned, unknown_counts = {}, Counter(), Counter()
        unattributed_rows = []
        for index, row in sorted(self.rows.items()):
            if not row["eligible"]:
                continue
            name, reason = _person(row["metadata"].get(column))
            if name is None:
                unassigned[reason] += 1
                unknown_counts["checked"] += row["checked"]
                unknown_counts["problem_rows"] += row["problem"]
                unattributed_rows.append({"excel_row": index + 1, "reason": reason})
                continue
            group = groups.setdefault(name, {"name": name, "records": 0, "checked": 0,
                "problem_rows": 0, "fixed_rows": 0, "review_rows": 0, "not_checked": 0,
                "source_names": [], "states": []})
            raw_name = row["metadata"][column].strip()
            if raw_name not in group["source_names"]:
                group["source_names"].append(raw_name)
            group["records"] += 1
            group["checked"] += row["checked"]
            group["problem_rows"] += row["problem"]
            group["fixed_rows"] += row["fixed"]
            group["review_rows"] += row["review"]
            group["not_checked"] += not row["checked"]
            if row["checked"]:
                group["states"].append(row["state"])
        ordered = sorted(groups.values(), key=lambda group: (
            group["checked"] == 0,
            -Fraction(group["problem_rows"], group["checked"] or 1),
            -group["problem_rows"], -group["checked"], group["name"]))
        prior_metric, rank = None, 0
        for index, group in enumerate(ordered, 1):
            metric = Fraction(group["problem_rows"], group["checked"] or 1)
            if metric != prior_metric:
                rank = index
            group["rank"] = rank if group["checked"] else None
            prior_metric = metric
            group["problem_rate"] = round(100 * float(metric), 1) if group["checked"] else None
            group["check_items"] = [item for item in aggregate_check_states(group.pop("states"))["items"] if item["affected"]]
        baseline = all(row["baseline"] for row in self.rows.values() if row["eligible"])
        basis_note = ("仅统计确定性底线规范问题。" if baseline else
                      "辅助筛查统计含模型疑似问题，不是已确认错误率，不作为底线结论。")
        return {"available": True, "version": 1, "column": column, "column_label": "标注员",
                "basis": "baseline" if baseline else "semantic_screening",
                "sort": "problem_rate_desc", "people": ordered,
                "attributed_records": sum(g["records"] for g in ordered),
                "checked": sum(g["checked"] for g in ordered),
                "unattributed_records": sum(unassigned.values()),
                "unattributed_checked": unknown_counts["checked"],
                "unattributed_problem_rows": unknown_counts["problem_rows"],
                "unattributed_reasons": dict(unassigned), "unattributed_rows": unattributed_rows,
                "note": basis_note + "问题涉及率 =（已生成修正或需处理的 Caption 数）÷ 已检 Caption 数。同率并列；已检量为 0 不排名。"}


def build_people_report(rows):
    tracker = PeopleTracker()
    for row in rows:
        tracker.add(row)
    return tracker.snapshot()


def people_report_path(output):
    return output.with_name(output.stem + "_人员问题排名.txt")


def export_people_report(report, output):
    if not report.get("available"):
        return None
    from captionqa.caption_qa_id_list import _write_atomic
    path = people_report_path(output)
    title = "底线规范问题" if report.get("basis") == "baseline" else "辅助筛查问题"
    lines = [f"Caption 人员{title}统计（按问题涉及率排序）", "人员取自标注员列，仅对 Caption 检测结果分组；人员字段不发送给模型。",
             report["note"], "多人归属不明和未填写人员的记录不纳入个人排序。", ""]
    for person in report["people"]:
        rate = f"{person['problem_rate']:.1f}%" if person["problem_rate"] is not None else "未检查"
        lines.append(f"{person['rank'] or '—'}. {person['name']}：已检 {person['checked']} 条，涉及问题 {person['problem_rows']} 条，涉及率 {rate}；"
                     f"已修复 {person['fixed_rows']} 条，需复核 {person['review_rows']} 条，未完成 {person['not_checked']} 条。")
        if person["check_items"]:
            lines.append("   检查项：" + "；".join(f"{item['label']} {item['affected']} 条" for item in person["check_items"]))
        if len(person["source_names"]) > 1 or person["source_names"] != [person["name"]]:
            lines.append("   合并写法：" + "、".join(person["source_names"]))
    lines += ["", f"人员归属待确认：{report['unattributed_records']} 条，其中已检 {report['unattributed_checked']} 条、涉及问题 {report['unattributed_problem_rows']} 条。"]
    for reason, count in report["unattributed_reasons"].items():
        lines.append(f"{reason}：{count} 条。")
    for item in report["unattributed_rows"]:
        lines.append(f"原表第 {item['excel_row']} 行：{item['reason']}。")
    _write_atomic(path, "\n".join(lines) + "\n")
    return str(path)
