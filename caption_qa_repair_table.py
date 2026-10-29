"""Two-column repair handoff, derived only from saved Caption QA findings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata

from caption_qa_summary import DISABLED


STATIC_LABELS = {
    "missing_punctuation": "缺少句末标点",
    "bare_tags": "标签格式不规范",
    "missing_subjects": "自然指令漏写已定义主体",
    "ghost_subjects": "引用未定义主体",
    "duplicate_defined_subjects": "主体重复定义",
    "skipped_numbering": "主体编号不连续",
    "symbol_format": "内容符号不成对",
    "symbol_width": "全角/半角符号",
    "label_separator": "多标签拼接符不规范",
}


def repair_report_path(output):
    output = Path(output)
    return output.with_name(output.stem + "_返修表.xlsx")


def _plain(value):
    # XML cannot carry most control characters; retain ordinary whitespace.
    return "".join(c for c in str(value) if c in "\n\t" or unicodedata.category(c) != "Cc")


def _issue_note(issue, sections):
    kind = issue.get("issue_type", "")
    section = issue.get("section", "")
    prefix = f"【{section}】" if section else ""
    snippet = issue.get("original_snippet", "")
    if kind == "missing_punctuation":
        # Locate the original paragraph without exporting its body or revision.
        positions = [str(index) for index, line in enumerate(
            [line.strip() for line in sections.get(section, "").splitlines() if line.strip()], 1)
            if line == snippet.strip()]
        location = f"第{'、'.join(positions)}段" if positions else "段尾"
        return f"{prefix}{location}缺少句末标点"
    if kind == "bare_tags":
        tag = re.search(r"(?:主体|场景|视频|图片|音频)\s*\d+", snippet)
        label = "标签内部多余空格" if "<" in snippet and ">" in snippet else "编号标签缺少尖括号"
        tag_name = re.sub(r'\s+', '', tag.group()) if tag else ''
        return f"{prefix}{label}" + (f"（{tag_name}）" if tag_name else "")
    if kind == "symbol_format":
        return f"{prefix}{issue.get('suggestion') or '内容符号不成对，请核对'}"
    if kind == "label_separator":
        return f"{prefix}{issue.get('suggestion') or '多标签拼接符不规范，请核对'}"
    if kind in {"missing_subjects", "ghost_subjects", "duplicate_defined_subjects", "skipped_numbering"}:
        tags = list(dict.fromkeys(re.findall(r"<主体\d+>", snippet)))
        return f"{prefix}{STATIC_LABELS[kind]}：{'、'.join(tags) if tags else snippet}（请核对）"
    label = {"无歧义错别字": "疑似错别字", "语法病句与语句不通顺": "疑似病句 / 语句不通顺",
             "隐式漏挂主体标签": "疑似隐式漏挂主体标签"}.get(kind, kind or "疑似内容问题")
    return f"{prefix}{label}：{issue.get('suggestion') or '请人工核对'}"


def read_repair_entries(cache, *, id_column, dry_run):
    from caption_qa_agent import parse_caption_sections
    cache = Path(cache)
    latest = {}
    with cache.open("rb") as stream:
        for line in stream:
            if line.strip() and line.endswith(b"\n"):
                row = json.loads(line)
                latest[row["row_index"]] = row
    resolutions = {}
    sidecar = cache.with_suffix(".id_resolutions.json")
    if sidecar.exists():
        document = json.loads(sidecar.read_text())
        if {r.get("qa_checkpoint", {}).get("source_sha256") for r in latest.values()} != {document.get("source_sha256")}:
            raise ValueError("ID 核对记录与输入文件不匹配，拒绝导出返修表。")
        resolutions = document.get("rows", {})
    grouped, missing = {}, []
    for index, row in sorted(latest.items()):
        if row.get("qa_status") == "SKIPPED_NOT_HUMAN_CAPTION" or row.get("qa_scope") == "input_error":
            continue
        checkpoint = row.get("qa_checkpoint", {})
        report = row.get("qa_report", {})
        if dry_run and row.get("qa_scope", report.get("qa_scope")) != "static_only":
            raise ValueError("底线返修表不能混入语义辅助结果。")
        caption = row.get(checkpoint.get("selected_caption_column") or checkpoint.get("caption_field"), "")
        try:
            sections = parse_caption_sections(caption)
        except ValueError:
            sections = {}
        notes, seen_subject_checks = [], set()
        for issue in report.get("issues", []):
            if issue.get("source") not in {"static", "llm"} or issue.get("issue_type") in DISABLED:
                continue
            if dry_run and (issue.get("source") != "static" or issue.get("issue_type") not in STATIC_LABELS):
                continue
            if issue.get("issue_type") in {"missing_subjects", "ghost_subjects", "duplicate_defined_subjects", "skipped_numbering"}:
                key = (issue.get("issue_type"), issue.get("section"))
                if key in seen_subject_checks:
                    continue
                seen_subject_checks.add(key)
                tags = [tag for candidate in report.get("issues", [])
                        if (candidate.get("issue_type"), candidate.get("section")) == key
                        and candidate.get("source") == issue.get("source")
                        for tag in re.findall(r"<主体\d+>", candidate.get("original_snippet", ""))]
                issue = dict(issue, original_snippet='|'.join(dict.fromkeys(tags)))
            note = _plain(_issue_note(issue, sections))
            if note not in notes:
                notes.append(note)
        if not dry_run and report.get("semantic_qa_performed") and report.get("need_human_review") and not notes:
            notes.append("需人工复核：" + _plain(report.get("human_review_reason") or "请核对语义辅助结果"))
        if not notes:
            continue
        identifier = row.get(id_column)
        if not (type(identifier) in (str, int) and str(identifier).strip() and not re.search(r"[\r\n]", str(identifier))):
            identifier = resolutions.get(str(index), {}).get("id")
        if not (type(identifier) in (str, int) and str(identifier).strip() and not re.search(r"[\r\n]", str(identifier))):
            missing.append(index + 1)
            continue
        bucket = grouped.setdefault(str(identifier), [])
        bucket.extend(note for note in notes if note not in bucket)
    entries = [[identifier, "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1))]
               for identifier, notes in grouped.items()]
    return entries, missing


def _artifact_runtime():
    root = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = root / "bin/node"
    dependencies = root / "node_modules"
    if not node.is_file() or not (dependencies / "@oai/artifact-tool").is_dir():
        raise ValueError("Excel 导出组件未就绪；检测结果和 ID 名单已保留，请恢复表格组件后重新导出。")
    return node, dependencies


def export_repair_table(cache, output, *, id_column, dry_run, complete, preview_path=None):
    entries, missing = read_repair_entries(cache, id_column=id_column, dry_run=dry_run)
    destination = repair_report_path(output)
    try:node, dependencies = _artifact_runtime()
    except ValueError:
        if preview_path:raise ValueError('当前安装可导出Excel；PNG预览需要可选的渲染组件。')
        from caption_qa_portable_exports import repair_xlsx
        repair_xlsx(destination,entries,dry_run,complete,missing)
        return {"repair_report_path":str(destination),"repair_report_rows":len(entries),"repair_report_missing_ids":missing}
    destination.parent.mkdir(parents=True, exist_ok=True)
    builder = Path(__file__).with_suffix(".mjs")
    # Bind only the supplied runtime dependencies; do not install into the repo.
    with tempfile.TemporaryDirectory(prefix="caption-repair-") as directory:
        work = Path(directory)
        (work / "node_modules").symlink_to(dependencies, target_is_directory=True)
        shutil.copyfile(builder, work / "build.mjs")
        payload = {"entries": entries, "dry_run": dry_run, "complete": complete and not missing,
                   "missing": missing, "preview_path": str(preview_path) if preview_path else None}
        (work / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        fd, name = tempfile.mkstemp(prefix=".repair-", suffix=".xlsx", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            subprocess.run([str(node), str(work / "build.mjs"), str(work / "data.json"), str(temporary)],
                           check=True, capture_output=True, timeout=120)
            temporary.replace(destination)
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError("返修表导出失败；检测缓存和 ID 名单已保留，可单独重试导出，无需重新检测。") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return {"repair_report_path": str(destination), "repair_report_rows": len(entries),
            "repair_report_missing_ids": missing}
