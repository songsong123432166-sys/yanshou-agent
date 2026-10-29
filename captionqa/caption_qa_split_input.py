"""Read explicitly named v2 Caption fields; never inspect report columns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

FIELDS = ("基调", "场景", "主体", "画面元素", "总结", "自然指令")
HEADERS = ("序列号", *(name + "v2" for name in FIELDS))
PREPARED_NAME = "caption_fields.jsonl"


def prepare_split_input(source, *, sheet_name=None, id_column=None):
    """Return None for other layouts. Retain original IDs, fields and provenance.

    Recognize only the complete, unambiguous v2 header contract. An explicitly
    selected Caption column is handled by the caller, without this fallback.
    """
    from captionqa.caption_qa_table_io import _openpyxl, _streaming_worksheet
    source = Path(source)
    if source.suffix.lower() != ".xlsx":
        return None
    book = _openpyxl().load_workbook(source, read_only=True, data_only=False)
    records, selected, skipped = [], [], []
    try:
        if sheet_name and sheet_name not in book.sheetnames:
            raise ValueError("指定的工作表不存在。未调用 API。")
        for name in ([sheet_name] if sheet_name else book.sheetnames):
            sheet = _streaming_worksheet(book, name)
            rows = sheet.iter_rows()
            cells = next(rows, ())
            headers = [c.value for c in cells]
            if set(HEADERS).issubset(headers) and not all(headers.count(h) == 1 for h in HEADERS):
                raise ValueError(f"{name} 的 Caption 分列标题重复，请明确保留一组字段。未调用 API。")
            if not all(headers.count(h) == 1 for h in HEADERS):
                skipped.append(name)
                continue
            if any(isinstance(h, str) and ("caption" in h.lower() or "人标" in h) for h in headers):
                # Prefer the existing explicit single-column selection flow.
                return None
            positions = [headers.index(h) for h in HEADERS]
            if id_column and id_column not in {"序列号", "A"}:
                raise ValueError("此分列表格的 ID 字段为“序列号”，请清空 ID 列设置或填写“序列号”。未调用 API。")
            if id_column == "A" and positions[0] != 0:
                raise ValueError("A 列不是序列号，未调用 API。")
            count = 0
            for number, row in enumerate(rows, 2):
                values = [row[i].value if i < len(row) else None for i in positions]
                if not any(v is not None and str(v).strip() for v in values[1:]):
                    continue  # Empty formatting/report-only rows are not Caption.
                if any(i < len(row) and row[i].data_type == "f" for i in positions):
                    raise ValueError(f"{name} 第 {number} 行的 Caption/序列号包含公式，请提供文本值。未调用 API。")
                if any(v is not None and not isinstance(v, str) for v in values[1:]):
                    raise ValueError(f"{name} 第 {number} 行的 Caption 字段不是文本。未调用 API。")
                identifier = values[0]
                if identifier is not None and not isinstance(identifier, (str, int)):
                    raise ValueError(f"{name} 第 {number} 行序列号格式无法无损读取。未调用 API。")
                caption = "".join(f"## {field}\n{value or ''}\n" for field, value in zip(FIELDS, values[1:]))
                from captionqa.caption_qa_agent import parse_caption_sections
                try:
                    parsed = parse_caption_sections(caption)
                    # Some exports retain a field's own label, e.g.
                    # 场景v2 cell = "场景：纯白背景。". The existing parser
                    # consumes that wrapper; allow only this exact same-field
                    # prefix, never a nested/renamed field in the body.
                    intact = list(parsed) == list(FIELDS) and all(
                        parsed[field].strip() in {
                            (value or '').strip(),
                            re.sub(r'^' + re.escape(field) + r'[：:][ \t]*', '', value or '').strip()}
                        for field, value in zip(FIELDS, values[1:]))
                except ValueError:
                    intact = False
                if not intact:
                    raise ValueError(f"{name} 第 {number} 行含嵌套字段标题，无法安全组合 Caption。未调用 API。")
                records.append({"id": identifier, "caption": caption,
                                "source_sheet": name, "source_row": number})
                count += 1
            selected.append({"name": name, "caption_rows": count})
    finally:
        book.close()
    if not selected:
        return None
    if not records:
        raise ValueError("分列 Caption 中没有可检查的正文。未调用 API。")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    target = source.parent / PREPARED_NAME
    # Stable bytes allow checkpoint reuse after service restart.
    if not target.exists() or target.read_text(encoding="utf-8") != payload:
        temporary = target.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sheets": selected, "ignored_sheets": skipped, "records": len(records),
                "caption_fields": list(FIELDS), "excluded_report_columns": True}
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": target, **manifest}
