"""Only the chosen Caption column and its content may enter QA."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook

import captionqa.caption_qa_agent as qa
import captionqa.caption_qa_web as web
from tests.test_caption_qa_sniff import CountingAgent, HUMAN, read_jsonl, write_jsonl
from tests.test_caption_qa_upgrades import GOOD, FakeAsyncClient, request_sections, semantic_reply


class CaptionScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "business.jsonl"
        self.target = self.root / "result.jsonl"

    def test_note_and_machine_columns_cannot_be_auto_selected(self):
        for column in ("质检备注", "备注", "comment", "qa_revised_caption", "机标caption", "模型caption"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "明确指定"):
                qa.detect_caption_column([{"caption": None, column: HUMAN}])

    def test_unknown_column_requires_explicit_selection(self):
        with self.assertRaisesRegex(ValueError, "明确指定"):
            qa.detect_caption_column([{"其他文本": HUMAN}])

    def test_noncaption_fields_never_enter_model_payload(self):
        write_jsonl(self.source, [{"id": "0001", "caption": GOOD,
                                  "质检备注": HUMAN.replace("白袍女子", "秘密备注内容"),
                                  "机器caption": HUMAN.replace("白袍女子", "秘密机标内容"),
                                  "标注员": "秘密人员信息"}])
        client = FakeAsyncClient(semantic_reply())
        agent = qa.CaptionQAAgent(async_client=client, max_retries=0)
        asyncio.run(qa.aprocess_batch(self.source, self.target, agent))
        self.assertEqual(len(client.calls), 1)
        payload = json.dumps(client.calls[0], ensure_ascii=False)
        for forbidden in ("秘密备注内容", "秘密机标内容", "秘密人员信息"):
            self.assertNotIn(forbidden, payload)
        row = read_jsonl(self.target)[0]
        self.assertEqual(row["qa_status"], "PASS")
        self.assertEqual(row["qa_checkpoint"]["selected_caption_column"], "caption")
        self.assertIn("秘密备注内容", row["质检备注"])

    def test_empty_caption_with_human_shaped_note_is_skipped(self):
        write_jsonl(self.source, [{"id": "0001", "caption": GOOD, "备注": HUMAN},
                                  {"id": "0002", "caption": "", "备注": HUMAN}])
        agent = CountingAgent()
        asyncio.run(qa.aprocess_batch(self.source, self.target, agent))
        self.assertEqual(agent.calls, [GOOD])
        row = next(row for row in read_jsonl(self.target) if row["id"] == "0002")
        self.assertEqual(row["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")
        self.assertIsNone(row["qa_checkpoint"]["selected_caption_column"])
        self.assertEqual(row["qa_issues_summary"], "")

    def test_compact_headers_and_noncontent_metadata_do_not_create_errors(self):
        metadata = "##任务类型\r\n视频1|主体9\r\n##similarity_label\r\n主体88<场景99>：<场景99>：\r\n"
        raw = metadata + GOOD.replace("## ", "##").replace("\n", "\r\n")
        client = FakeAsyncClient(semantic_reply())
        report = asyncio.run(qa.CaptionQAAgent(async_client=client, max_retries=0).acheck_single_caption(raw))
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["semantic_qa_performed"])
        self.assertEqual(report["revised_caption"], raw)
        self.assertEqual(report["issues"], [])
        self.assertNotIn("similarity_label", request_sections(client.calls[0]))

    def test_old_scope_checkpoint_cannot_resume_as_current_result(self):
        write_jsonl(self.source, [{"id": "0001", "caption": GOOD}])
        asyncio.run(qa.aprocess_batch(self.source, self.target, CountingAgent()))
        row = read_jsonl(self.target)[0]
        row["qa_checkpoint"]["human_caption_filter"] = 1
        self.target.write_text(json.dumps(row) + "\n")
        agent = CountingAgent()
        with self.assertRaisesRegex(ValueError, "不匹配"):
            asyncio.run(qa.aprocess_batch(self.source, self.target, agent))
        self.assertEqual(agent.calls, [])

    def test_excel_secondary_caption_header_ignores_caption_shaped_notes(self):
        source = self.source.with_suffix(".xlsx")
        book = Workbook()
        sheet = book.active
        sheet.append(["样本ID", "人工标注", None, "人工质检"])
        sheet.append([None, "是否运镜", "caption", "质检备注"])
        sheet.append(["0001", "是", GOOD, HUMAN])
        sheet.append(["0002", "是", None, HUMAN])
        book.save(source)
        book.close()
        self.assertEqual(qa.detect_caption_column(source), "__column_3")
        agent = CountingAgent()
        asyncio.run(qa.aprocess_batch(source, self.target, agent))
        self.assertEqual(agent.calls, [GOOD])
        self.assertEqual(next(row for row in read_jsonl(self.target) if row["row_index"] == 3)["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")

    def test_csv_secondary_caption_header_ignores_notes_and_machine_text(self):
        source = self.source.with_suffix(".csv")
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows([
                ["样本ID", "人工标注", "", "机器标注参考", "人工质检"],
                ["", "是否运镜", "caption", "机标caption", "质检备注"],
                ["0001", "是", GOOD, HUMAN, HUMAN],
                ["0002", "是", "", HUMAN, HUMAN],
            ])
        original = source.read_bytes()
        self.assertEqual(qa.detect_caption_column(source), "__column_3")
        agent = CountingAgent()
        output = self.root / "ids.txt"
        asyncio.run(qa.aprocess_batch(source, output, agent))
        self.assertEqual(agent.calls, [GOOD])
        self.assertEqual(output.read_text(), "")
        self.assertEqual(source.read_bytes(), original)
        rows = sorted(read_jsonl(output.with_suffix(".jsonl")), key=lambda row: row["row_index"])
        self.assertEqual(rows[0]["qa_skip_reason"], "table_header")
        self.assertEqual(rows[2]["qa_skip_reason"], "empty_caption")
        self.assertEqual(json.loads(output.with_suffix(".summary.json").read_text())["skipped_reasons"],
                         {"多级表头行": 1, "Caption 为空": 1})

    def test_csv_secondary_header_cannot_override_excluded_column(self):
        source = self.source.with_suffix(".csv")
        with source.open("w", newline="") as stream:
            csv.writer(stream).writerows([["样本ID", "质检备注", "机标caption"],
                                         ["", "caption", "人标caption"], ["0001", GOOD, HUMAN]])
        with self.assertRaisesRegex(ValueError, "明确指定"):
            qa.detect_caption_column(source)

    def test_later_csv_cell_cannot_reclassify_an_unknown_column(self):
        source = self.source.with_suffix(".csv")
        with source.open("w", newline="") as stream:
            csv.writer(stream).writerows([["样本ID", "其他文本"], ["0001", "说明"],
                                         ["0002", "caption"], ["0003", GOOD]])
        with self.assertRaisesRegex(ValueError, "明确指定"):
            qa.detect_caption_column(source)

    def test_web_csv_secondary_header_completes_with_readable_column_location(self):
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows([["样本ID", "人工标注", ""],
                                     ["", "是否运镜", "caption"], ["0001", "是", GOOD]])
        data = stream.getvalue().encode()
        app = web.App(self.root / "Downloads", self.root / "work")
        self.addCleanup(app.close)
        with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
            item = app.create_job("business.csv", io.BytesIO(data), len(data), review_first=False, dry_run=True)
            app.executor.submit(lambda: None).result(timeout=10)
        job = app.jobs[item["id"]].snapshot()
        self.assertEqual(job["status"], "done", job["error"])
        self.assertEqual(job["caption_location"], "CSV 第 3 列（Caption）")
        self.assertEqual(job["processed"], 2)
        self.assertEqual(job["counts"], {"SKIPPED_NOT_HUMAN_CAPTION": 1, "PASS": 1})

    def test_note_only_worksheet_is_not_selected(self):
        source = self.source.with_suffix(".xlsx")
        book = Workbook()
        book.active.append(["样本ID", "质检备注"])
        book.active.append(["0001", HUMAN])
        book.save(source)
        book.close()
        app = web.App(self.root / "Downloads", self.root / "work")
        self.addCleanup(app.close)
        data = source.read_bytes()
        with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
            job = app.create_job("business.xlsx", io.BytesIO(data), len(data), review_first=False)
            app.executor.submit(lambda: None).result(timeout=10)
        self.assertEqual(app.jobs[job["id"]].status, "failed")

    def test_stale_history_is_hidden_and_retry_preserves_old_cache(self):
        app = web.App(self.root / "Downloads", self.root / "work")
        self.addCleanup(app.close)
        data = (json.dumps({"id": "0001", "caption": GOOD}) + "\n").encode()
        item = app.create_job("business.jsonl", io.BytesIO(data), len(data), review_first=False, dry_run=True)
        app.executor.submit(lambda: None).result(timeout=10)
        job = app.jobs[item["id"]]
        old_cache = job.cache
        app.save_history()
        row = json.loads(old_cache.read_text())
        row["qa_checkpoint"]["human_caption_filter"] = 1
        old_cache.write_text(json.dumps(row) + "\n")
        original_bytes = old_cache.read_bytes()
        with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
            restored = web.App(app.output_dir, app.work_dir)
            self.addCleanup(restored.close)
            job = restored.jobs[item["id"]]
            self.assertTrue(job.stale_results)
            self.assertEqual(job.snapshot()["processed"], 0)
            self.assertFalse(job.snapshot()["output_ready"])
            restored.retry(job.id)
            restored.executor.submit(lambda: None).result(timeout=10)
        self.assertEqual(job.status, "done", job.error)
        self.assertNotEqual(job.cache, old_cache)
        self.assertEqual(old_cache.read_bytes(), original_bytes)
        self.assertEqual(json.loads(job.cache.read_text())["qa_checkpoint"]["human_caption_filter"], qa.CAPTION_SCOPE_VERSION)

    def test_explicit_note_column_is_rejected_before_requests(self):
        write_jsonl(self.source, [{"id": "0001", "质检备注": HUMAN}])
        agent = CountingAgent()
        with self.assertRaisesRegex(ValueError, "备注"):
            asyncio.run(qa.aprocess_batch(self.source, self.target, agent, column="质检备注"))
        self.assertEqual(agent.calls, [])
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
