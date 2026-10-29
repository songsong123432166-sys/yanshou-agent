"""Editable, source-bound review tables with red excerpt runs and subject data."""
import copy
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from caption_qa_repair_table import _artifact_runtime, repair_report_path
from caption_qa_review_state import decide, human_review_issues, automatic_issue_ids
from caption_qa_table_io import iter_table_records

HEADERS = ["错误项编号", "题目 ID", "字段", "错误标签", "原文节选", "主体字段（原文）", "审核结论", "人工修改后的完整字段"]
from caption_qa_review_model import DECISION_LABELS
STATUS = DECISION_LABELS


def highlight_ranges(text, issue):
    if issue['kind']=='symbol_width' and 'excerpt_highlights' in issue:
        if text==issue.get('subject') and issue.get('subject_highlights'):
            return issue['subject_highlights']
        if text=='\n'.join(issue['excerpts']):
            spans=[];offset=0
            for excerpt,ranges in zip(issue['excerpts'],issue.get('excerpt_highlights',[])):
                spans.extend([a+offset,b+offset] for a,b in ranges)
                offset+=len(excerpt)+1
            return spans
        return []
    snippet = issue["snippet"].strip()
    parts = re.findall(r"<主体\d+>", snippet) if issue["kind"] in {"missing_subjects", "ghost_subjects", "skipped_numbering", "duplicate_defined_subjects"} else [snippet]
    if not any(p and p in text for p in parts):
        parts = [p.strip() for p in re.split(r"[。！？!?\n]", snippet) if len(p.strip()) > 1]
    spans = [(m.start(), m.end()) for p in parts if p for m in re.finditer(re.escape(p), text)]
    result = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def patch_rich_text(path, highlights):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    tag = lambda key: "{" + ns + "}" + key
    ET.register_namespace("", ns)
    ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    path = Path(path)
    with ZipFile(path) as zin:
        root = ET.fromstring(zin.read("xl/worksheets/sheet1.xml"))
        cells = {c.get("r"): c for c in root.iter(tag("c"))}
        for address, (text, spans) in highlights.items():
            if not spans:
                continue
            cell = cells[address]
            for child in list(cell):
                cell.remove(child)
            cell.set("t", "inlineStr")
            inline = ET.SubElement(cell, tag("is"))
            bounds = sorted({0, len(text), *(n for span in spans for n in span)})
            for a, b in zip(bounds, bounds[1:]):
                run = ET.SubElement(inline, tag("r"))
                if any(x <= a and b <= y for x, y in spans):
                    props = ET.SubElement(run, tag("rPr"))
                    ET.SubElement(props, tag("b"))
                    ET.SubElement(props, tag("color"), {"rgb": "FFCC0000"})
                ET.SubElement(run, tag("t"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}).text = text[a:b]
        raw = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        temporary = path.with_suffix(".rich.tmp")
        with ZipFile(temporary, "w") as zout:
            for info in zin.infolist():
                zout.writestr(info, raw if info.filename == "xl/worksheets/sheet1.xml" else zin.read(info))
    temporary.replace(path)


def export_review(document, output, preview_path=None):
    output = Path(output)
    destination = output.with_name(output.stem + f"_人工审核_v{document['revision']}_{uuid4().hex[:6]}.xlsx")
    rows, highlights = [], {}
    visible=human_review_issues(document)
    for index, issue in enumerate(visible, 2):
        excerpt = "\n".join(issue["excerpts"])
        final = document["field_edits"].get(issue["field_id"], "")
        note=f"【{issue['label']}】{issue['reason']}"
        if issue.get('requires_confirmation'):
            note+='\n候选替换（需人工确认）：'+'；'.join(dict.fromkeys(f"{op.get('candidate_old',op['old'])} → {op.get('candidate_new',op['new'])}" for op in issue['operations']))
        rows.append([issue["issue_id"], issue["sample_id"], issue["field"],
                     note, excerpt, issue["subject"], STATUS[issue["decision"]], final])
        highlights[f"E{index}"] = (excerpt, highlight_ranges(excerpt, issue))
        if issue["subject"]:
            highlights[f"F{index}"] = (issue["subject"], highlight_ranges(issue["subject"], issue))
    if any(len(value) > 32767 for row in rows for value in row):
        raise ValueError("审核字段超过 Excel 单元格上限，请在网页审核。")
    csv_path = destination.with_name(destination.stem + "_可回传.csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    try:node, dependencies = _artifact_runtime()
    except ValueError:
        if preview_path:raise ValueError('当前安装可导出Excel；PNG预览需要可选的渲染组件。')
        from caption_qa_portable_exports import review_xlsx
        review_xlsx(destination,HEADERS,rows,highlights)
        return {"repair_report_path":str(destination),"review_csv_path":str(csv_path),"repair_report_rows":len(rows),
                "repair_report_missing_ids":[i["row_index"]+1 for i in visible if not i["sample_id"]]}
    with tempfile.TemporaryDirectory(prefix="caption-review-report-") as directory:
        work = Path(directory)
        (work / "node_modules").symlink_to(dependencies, target_is_directory=True)
        shutil.copyfile(Path(__file__).with_suffix(".mjs"), work / "build.mjs")
        (work / "data.json").write_text(json.dumps({"headers": HEADERS, "rows": rows, "decision_labels":list(DECISION_LABELS.values())}, ensure_ascii=False))
        temporary = work / "report.xlsx"
        try:
            subprocess.run([str(node), str(work / "build.mjs"), str(work / "data.json"), str(temporary)], check=True, capture_output=True, timeout=120)
            patch_rich_text(temporary, highlights)
            if preview_path:
                # Rendering is done after the final rich-text file is imported.
                subprocess.run([str(node), str(work / "build.mjs"), "--preview", str(temporary), str(preview_path)], check=True, capture_output=True, timeout=120)
            shutil.copyfile(temporary, destination)
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError("Excel 审核表导出失败；可在网页审核，或使用已保存的可回传 CSV。") from exc
    return {"repair_report_path": str(destination), "review_csv_path": str(csv_path), "repair_report_rows": len(rows),
            "repair_report_missing_ids": [i["row_index"] + 1 for i in visible if not i["sample_id"]]}


def import_review(document, path, *, revision):
    if revision != document["revision"]:
        raise ValueError("审核版本已变化，请刷新后重新导入。")
    result = copy.deepcopy(document)
    known = {i["issue_id"]: i for i in result["issues"]}
    fields = {f["field_id"]: f for f in result["fields"]}
    automatic=automatic_issue_ids(result)
    labels = {value: key for key, value in STATUS.items()}
    records = list(iter_table_records(path))
    incoming, final_fields = {}, {}
    for _, row, error in records:
        if error:
            raise ValueError("审核表有无法解析的行，未导入任何修改。")
        if not any(v is not None and str(v).strip() for v in row.values()):
            continue
        if not set(HEADERS).issubset(row):
            raise ValueError("审核表列名不完整，请使用导出的审核模板。")
        key = str(row[HEADERS[0]] or "")
        if key not in known or key in incoming:
            raise ValueError("错误项编号未知或重复，请勿修改编号或导入其他任务的审核表。")
        issue = known[key]
        if str(row[HEADERS[1]] or "") != issue["sample_id"] or row[HEADERS[2]] != issue["field"]:
            raise ValueError("题目 ID 或字段与检测原文不一致，未导入。")
        if (row[HEADERS[4]] != "\n".join(issue["excerpts"]) or (row[HEADERS[5]] or "") != issue["subject"]):
            raise ValueError("原文节选或主体原文已被改动；请只编辑审核结论和人工修改后的完整字段。")
        decision = labels.get(str(row[HEADERS[6]] or "").strip())
        if decision is None:
            raise ValueError("审核结论不支持；请使用模板中的审核选项。")
        if decision == 'auto' and issue['decision'] != 'auto':
            raise ValueError('不能将其他问题标为自动修正；请人工确认指定改法或填写完整定稿。')
        text = row[HEADERS[7]]
        if text is not None and not isinstance(text, str):
            raise ValueError("人工改稿必须是完整文本。")
        if text and decision != "manual":
            raise ValueError("填写人工改稿后，请将审核结论设为“已人工修改”。")
        if text:
            field_id = issue["field_id"]
            if field_id in final_fields and final_fields[field_id] != text:
                raise ValueError("同一字段出现多份不同人工稿，请统一后导入。")
            final_fields[field_id] = text
        incoming[key] = decision
    # An import is a complete review snapshot, not an implicit approval of
    # missing rows. Incomplete subsets are accepted only outside manual fields.
    for field_id in final_fields:
        if any(incoming.get(key) not in {"manual", "reject", "keep", "keep_highlight", "auto"} for key in fields[field_id]["issue_ids"] if key not in automatic):
            raise ValueError("人工定稿字段仍有未审核或不同执行方式的项目，请统一该字段的审核结论。")
    for key, decision in incoming.items():
        if known[key]["field_id"] in final_fields:
            continue
        if decision == "manual":
            raise ValueError("已人工修改的字段未提供完整改稿。")
        if decision == 'auto':
            continue
        result = decide(result, revision=result["revision"], issue_id=key, decision=decision)
    for field_id, text in final_fields.items():
        for key in fields[field_id]["issue_ids"]:
            if incoming.get(key) in {"reject", "keep", "keep_highlight"}:
                result["issues"][next(n for n,i in enumerate(result["issues"]) if i["issue_id"] == key)]["decision"] = incoming[key]
        result = decide(result, revision=result["revision"], field_id=field_id, final_text=text)
    return result
