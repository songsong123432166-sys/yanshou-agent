"""Problem IDs exclude skipped rows and operational/format failures."""
from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import caption_qa_agent as qa
from caption_qa_id_list import _pending_content_notes, detect_id_column
from test_caption_qa_sniff import CountingAgent, read_jsonl, write_jsonl
from test_caption_qa_upgrades import GOOD, FakeAsyncClient, semantic_issue, semantic_reply


class IdListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "business.jsonl"
        self.output = self.root / "问题ID名单.txt"

    def summary(self):
        return json.loads(self.output.with_suffix(".summary.json").read_text())

    def test_every_run_outputs_only_problem_ids_and_can_resume(self):
        bad = GOOD.replace("身穿白袍。", "身穿白袍")
        records = [{"id": "0001", "caption": GOOD, "备注": bad},
                   {"id": "0002", "caption": bad}, {"id": "0003", "caption": ""},
                   {"id": "0004", "caption": "不是支持的分段格式"}]
        write_jsonl(self.source, records)
        original = self.source.read_bytes()
        agent = CountingAgent()
        asyncio.run(qa.aprocess_batch(self.source, self.output, agent))
        self.assertEqual(self.output.read_text(), "0002\n")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(list(self.root.glob("*.xlsx")))
        self.assertFalse(list(self.root.glob("*.csv")))
        self.assertEqual(self.summary()["skipped_reasons"], {"Caption 为空": 1, "不符合支持的分段 Caption 格式": 1})
        resumed = CountingAgent()
        self.assertEqual(asyncio.run(qa.aprocess_batch(self.source, self.output, resumed)), 0)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(self.output.read_text(), "0002\n")

    def test_api_failure_is_not_a_content_problem_id(self):
        write_jsonl(self.source, [{"id": "0001", "caption": GOOD}])
        client = FakeAsyncClient(ValueError("offline"))
        agent = qa.CaptionQAAgent(async_client=client, max_retries=0)
        asyncio.run(qa.aprocess_batch(self.source, self.output, agent))
        self.assertEqual(self.output.read_text(), "")
        self.assertEqual(self.summary()["not_checked"], 1)
        self.assertFalse(self.summary()["complete"])
        self.assertIn("不是完整结论", self.output.with_name("问题ID名单_检测说明.txt").read_text())

    def test_fixed_text_is_omitted_from_explanation_but_id_and_counts_are_retained(self):
        raw = GOOD.replace("身穿白袍。", "身穿独特银灰色礼服")
        write_jsonl(self.source, [{"id": "0001", "caption": raw}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        explanation = self.output.with_name("问题ID名单_检测说明.txt").read_text()
        self.assertEqual(self.output.read_text(), "0001\n")
        self.assertNotIn("独特银灰色礼服", explanation)
        self.assertNotIn("0001（原表第", explanation)
        self.assertIn("无待复核的内容问题", explanation)
        self.assertIn("标点问题：涉及 1 条，已修复 1 条，需复核 0 条", explanation)
        self.assertIn("独特银灰色礼服", read_jsonl(self.output.with_suffix(".jsonl"))[0]["qa_report"]["revised_caption"])

    def test_mixed_record_only_lists_unresolved_issue(self):
        raw = GOOD.replace("身穿白袍。", "身穿只用于已修复项的衣服")
        write_jsonl(self.source, [{"id": "0001", "caption": raw}])
        answer = semantic_reply(issues=[semantic_issue("总结", "奔跑", "人工核对句式", "语法病句与语句不通顺")])
        agent = qa.CaptionQAAgent(async_client=FakeAsyncClient(answer), max_retries=0)
        asyncio.run(qa.aprocess_batch(self.source, self.output, agent))
        explanation = self.output.with_name("问题ID名单_检测说明.txt").read_text()
        self.assertIn("0001（原表第 2 行）", explanation)
        self.assertIn("人工核对句式", explanation)
        self.assertNotIn("只用于已修复项的衣服", explanation)
        self.assertNotIn("在段落末尾补全句号", explanation)
        self.assertEqual(self.output.read_text(), "0001\n")

    def test_review_reason_without_issue_still_appears(self):
        write_jsonl(self.source, [{"id": "0001", "caption": GOOD}])
        answer = semantic_reply(need_human_review=True, human_review_reason="句式存在歧义，请人工核对")
        asyncio.run(qa.aprocess_batch(self.source, self.output,
            qa.CaptionQAAgent(async_client=FakeAsyncClient(answer), max_retries=0)))
        explanation = self.output.with_name("问题ID名单_检测说明.txt").read_text()
        self.assertIn("句式存在歧义", explanation)
        self.assertIn("请人工核对", explanation)

    def test_guard_does_not_export_entire_already_revised_paragraph(self):
        row = {"qa_report": {"need_human_review": True, "issues": [
            {"source": "merge", "section": "自然指令", "issue_type": "unsafe_llm_edit",
             "resolved": False, "original_snippet": "已经补过标点的整段内容，不应出现在说明中。",
             "suggestion": "订正稿改变了数值结构，请人工核对"}]}}
        note = _pending_content_notes(row)
        self.assertIn("修订未应用", note)
        self.assertIn("数值结构", note)
        self.assertNotIn("已经补过标点的整段内容", note)

    def test_flattened_summary_never_reintroduces_fixed_content(self):
        row = {"qa_issues_summary": "旧汇总中的已修复正文不应回流", "qa_report": {
            "need_human_review": False, "issues": [{"resolved": True,
                "original_snippet": "已修复原文", "suggestion": "已修复建议"}]}}
        self.assertEqual(_pending_content_notes(row), "")

    def test_parser_failure_is_separate_from_content_problem_ids(self):
        invalid = GOOD + "## 主体\n重复的结构字段。\n"
        write_jsonl(self.source, [{"id": "0001", "caption": invalid}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        self.assertEqual(self.output.read_text(), "")
        self.assertEqual(self.summary()["not_checked"], 1)

    def test_missing_id_is_not_replaced_by_an_excel_row_number(self):
        bad = GOOD.replace("身穿白袍。", "身穿白袍")
        write_jsonl(self.source, [{"id": "", "caption": bad}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        self.assertEqual(self.output.read_text(), "")
        self.assertEqual(self.summary()["missing_problem_ids"], [2])
        self.assertFalse(self.summary()["complete"])

    def test_verified_id_resolution_only_fills_missing_id_and_survives_resume(self):
        bad = GOOD.replace("身穿白袍。", "身穿白袍")
        write_jsonl(self.source, [{"id": "", "caption": bad}, {"id": "0002", "caption": bad}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        row = read_jsonl(self.output.with_suffix(".jsonl"))[0]
        resolution = {"source_sha256": row["qa_checkpoint"]["source_sha256"],
                      "rows": {"1": {"id": "0001", "provenance": "同工作簿唯一匹配记录"},
                               "2": {"id": "wrong", "provenance": "不能覆盖已有ID"}}}
        self.output.with_suffix(".id_resolutions.json").write_text(json.dumps(resolution))
        agent = CountingAgent()
        asyncio.run(qa.aprocess_batch(self.source, self.output, agent))
        self.assertEqual(agent.calls, [])
        self.assertEqual(self.output.read_text(), "0001\n0002\n")
        self.assertTrue(self.summary()["complete"])
        self.assertEqual(len(self.summary()["resolved_ids"]), 1)

    def test_id_resolution_from_another_source_is_rejected(self):
        bad = GOOD.replace("身穿白袍。", "身穿白袍")
        write_jsonl(self.source, [{"id": "", "caption": bad}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        self.output.with_suffix(".id_resolutions.json").write_text(json.dumps({"source_sha256": "other", "rows": {}}))
        with self.assertRaisesRegex(ValueError, "不匹配"):
            asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))

    def test_unknown_id_column_fails_before_requests(self):
        write_jsonl(self.source, [{"caption": GOOD, "其他": "not-an-id"}])
        agent = CountingAgent()
        with self.assertRaisesRegex(ValueError, "ID 列"):
            asyncio.run(qa.aprocess_batch(self.source, self.output, agent))
        self.assertEqual(agent.calls, [])

    def test_sample_ids_are_preferred_over_video_ids(self):
        write_jsonl(self.source, [{"video_id": "video-1", "样本ID": "0001", "caption": GOOD}])
        self.assertEqual(detect_id_column(self.source), "样本ID")

    def test_export_with_broken_id_header_and_whitespace_rows(self):
        write_jsonl(self.source, [{"=SUM()": "cm77b763cb", "caption": GOOD},
                                  {"=SUM()": " ", "caption": ""},
                                  {"=SUM()": "cmd043a5a9", "caption": GOOD}])
        self.assertEqual(detect_id_column(self.source), "=SUM()")

    def test_duplicate_sample_id_is_listed_once(self):
        bad = GOOD.replace("身穿白袍。", "身穿白袍")
        write_jsonl(self.source, [{"样本ID": "0001", "caption": bad}, {"样本ID": "0001", "caption": bad}])
        asyncio.run(qa.aprocess_batch(self.source, self.output, CountingAgent()))
        self.assertEqual(self.output.read_text(), "0001\n")
        self.assertEqual(self.summary()["problem_ids"], 1)

    def test_cli_defaults_to_txt_instead_of_a_result_table(self):
        source = self.root / "business.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "caption"])
            writer.writerow(["0001", GOOD])
            writer.writerow(["0002", GOOD.replace("身穿白袍。", "身穿白袍")])
        result = subprocess.run([sys.executable, qa.__file__, "--dry-run", "--input", str(source)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(source.with_name("business_问题ID名单.txt").read_text(), "0002\n")
        self.assertFalse(source.with_name("business_qa_result.csv").exists())

    def test_caption_wrapper_and_compact_headers_preserve_original_text(self):
        for prefix in ("batch1-58\n", '"', "\ufeff"):
            raw = prefix + "##任务类型\n人物参考\n" + GOOD.replace("## ", "##")
            report = qa.CaptionQAAgent(dry_run=True).check(raw)
            self.assertEqual(report["status"], "PASS", prefix)
            self.assertEqual(report["revised_caption"], raw)

    def test_plain_and_mixed_section_headings_are_checked_without_reformatting(self):
        for text in (GOOD.replace("## ", ""), GOOD.replace("## 主体", "主体"),
                     GOOD.replace("## 自然指令", "自然指令（细节）")):
            self.assertTrue(qa.is_human_caption(text))
            report = qa.CaptionQAAgent(dry_run=True).check(text)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["revised_caption"], text)

    def test_plain_metadata_stays_outside_checks(self):
        text = "任务类型\n视频1 主体9\nsimilarity_label\n主体88\n" + GOOD.replace("## ", "")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], text)

    def test_inline_instruction_and_empty_final_heading_are_not_format_errors(self):
        for text in (GOOD.replace("## 自然指令\n", "## 自然指令    "),
                     GOOD + "画面元素   "):
            self.assertTrue(qa.is_human_caption(text))
            report = qa.CaptionQAAgent(dry_run=True).check(text)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["revised_caption"], text)

    def test_repeated_empty_heading_wrapper_is_preserved_without_false_alarm(self):
        text = GOOD.replace("## 总结\n", "## 总结\n总结\n")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], text)


if __name__ == "__main__":
    unittest.main()
