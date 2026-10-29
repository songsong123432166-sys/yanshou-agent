"""Offline acceptance tests for human-caption sniffing and pre-request filtering."""
from __future__ import annotations

import asyncio
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import caption_qa_agent as qa


HUMAN = "## 任务类型\n人物参考\n## 主体\n<主体1>：白袍女子。\n## 自然指令\n<主体1>向前奔跑。\n"
OTHER_HUMAN = HUMAN.replace("白袍女子", "黑衣男子")
MACHINE = "一名身穿白色长袍的年轻女子在昏暗山洞中向前奔跑，背景中有许多人。"
QA_COLUMNS = ("qa_status", "qa_missing_or_dup_subjects", "qa_issues_summary",
              "qa_human_review_reason", "qa_revised_caption")


class CountingAgent:
    """Fake transport with real static/assembly code; never constructs an API client."""

    def __init__(self) -> None:
        self.dry_run = True
        self.llm_options: dict[str, Any] = {}
        self.calls: list[str] = []
        self.closed = False

    async def acheck_single_caption(self, caption: str) -> qa.QAReport:
        self.calls.append(caption)
        await asyncio.sleep(0)
        return qa.CaptionQAAgent(dry_run=True).check(caption)

    @staticmethod
    def failure_report(caption: str, reason: str) -> qa.QAReport:
        return qa.CaptionQAAgent.failure_report(caption, reason)

    async def aclose(self) -> None:
        self.closed = True


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class HumanCaptionTests(unittest.TestCase):
    def test_nonstring_and_short_values_are_rejected(self) -> None:
        for value in (None, 0, True, float("nan"), {}, [], "", " " * 100,
                      "## 主体\n## 总结\n## 提示"):
            with self.subTest(value=repr(value)):
                self.assertFalse(qa.is_human_caption(value))

    def test_three_distinct_anchors_and_subject_or_instruction_are_required(self) -> None:
        self.assertTrue(qa.is_human_caption(HUMAN))
        self.assertTrue(qa.is_human_caption("## 提示\n生成一个人物视频。\n## 基调\n写实电影风格。\n## 自然指令\n人物向前奔跑。"))
        self.assertFalse(qa.is_human_caption("## 提示\n生成一个人物视频。\n## 基调\n写实电影风格。\n## 总结\n人物向前奔跑。"))
        self.assertFalse(qa.is_human_caption("## 主体\n<主体1>：白袍女子。\n## 总结\n<主体1>奔跑。"))

    def test_repeated_headers_do_not_inflate_anchor_count(self) -> None:
        self.assertFalse(qa.is_human_caption("## 主体\n女子穿白袍。\n" * 4))

    def test_compact_headers_and_crlf_are_recognized(self) -> None:
        caption = "##任务类型  \r\n人物参考\r\n##主体\r\n<主体1>：白袍女子。\r\n##自然指令 \r\n<主体1>向前奔跑。"
        self.assertTrue(qa.is_human_caption(caption))

    def test_plain_text_and_non_header_mentions_are_rejected(self) -> None:
        self.assertFalse(qa.is_human_caption(MACHINE))
        self.assertFalse(qa.is_human_caption("以下是字段说明：## 主体、## 自然指令、## 总结。请填写以上所有字段。"))


class DetectCaptionColumnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_best_hit_count_ignores_misleading_column_names_and_logs_selection(self) -> None:
        records = [{"caption": MACHINE, "人工审核文本": HUMAN, "补充": HUMAN},
                   {"caption": MACHINE, "人工审核文本": OTHER_HUMAN, "补充": ""}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = qa.detect_caption_column(records)
        self.assertEqual(result, "人工审核文本")
        self.assertIn("[Auto-Detect]", out.getvalue())
        self.assertIn("人工审核文本", out.getvalue())
        self.assertIn("有效人标行数: 2 / 总行数: 2", out.getvalue())

    def test_ambiguous_caption_columns_require_selection(self) -> None:
        records = [{"人标A": HUMAN, "人标B": OTHER_HUMAN}]
        with self.assertRaisesRegex(ValueError, "多个 Caption"):
            qa.detect_caption_column(records)

    def test_no_human_caption_raises_clear_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "人标|Caption|caption"):
            qa.detect_caption_column([{"caption": MACHINE}, {"caption": None}])

    def test_empty_rows_do_not_consume_nonempty_sample_budget(self) -> None:
        records = [{"人工": None, "机标": ""} for _ in range(55)]
        records.append({"人工": HUMAN, "机标": MACHINE})
        self.assertEqual(qa.detect_caption_column(records), "人工")

    def test_sample_limit_selects_from_first_fifty_when_they_have_hits(self) -> None:
        records = [{"人标A": HUMAN, "人标B": MACHINE} for _ in range(50)]
        records.extend({"人标A": MACHINE, "人标B": OTHER_HUMAN} for _ in range(60))
        self.assertEqual(qa.detect_caption_column(records), "人标A")
        with self.assertRaisesRegex(ValueError, "多个 Caption"):
            qa.detect_caption_column(records, sample_size=110)

    def test_zero_hit_initial_sample_can_discover_late_human_rows(self) -> None:
        records = [{"人工": MACHINE} for _ in range(50)] + [{"人工": HUMAN}]
        self.assertEqual(qa.detect_caption_column(records), "人工")

    def test_csv_path_detection_preserves_unicode_headers_and_multiline_cells(self) -> None:
        source = self.root / "business.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "url", "人标 Caption", "机标caption"])
            writer.writeheader()
            writer.writerow({"video_id": "00017", "url": "https://example.invalid/17",
                             "人标 Caption": HUMAN, "机标caption": MACHINE})
        self.assertEqual(qa.detect_caption_column(source), "人标 Caption")

    def test_jsonl_path_detection_scans_late_appearing_columns(self) -> None:
        source = self.root / "business.jsonl"
        write_jsonl(source, [{"id": 1, "machine": MACHINE}, {"id": 2, "后出现的人标列": HUMAN}])
        self.assertEqual(qa.detect_caption_column(source), "后出现的人标列")


class SniffedBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "business.jsonl"
        self.target = Path(self.directory.name) / "qa_cache.jsonl"

    async def test_autodetect_locks_caption_column_and_preserves_other_columns(self) -> None:
        records = [
            {"id": "primary1", "video_id": "00001", "人标": HUMAN, "备用": "", "机标caption": MACHINE},
            {"id": "primary2", "video_id": "00002", "人标": OTHER_HUMAN, "备用": "", "机标caption": MACHINE},
            {"id": "fallback", "video_id": "00003", "人标": MACHINE, "备用": HUMAN, "机标caption": MACHINE},
            {"id": "machine", "video_id": "00004", "人标": MACHINE, "备用": "", "机标caption": MACHINE},
            {"id": "blank", "video_id": "00005", "人标": None, "备用": "", "机标caption": ""},
        ]
        write_jsonl(self.source, records)
        agent = CountingAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent, concurrency=2), 5)
        self.assertCountEqual(agent.calls, [HUMAN, OTHER_HUMAN])
        rows = {row["id"]: row for row in read_jsonl(self.target)}
        for record in records:
            for key, value in record.items():
                self.assertEqual(rows[record["id"]][key], value)
            self.assertTrue(set(QA_COLUMNS).issubset(rows[record["id"]]))
        for identifier in ("fallback", "machine", "blank"):
            self.assertEqual(rows[identifier]["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")
            self.assertTrue(rows[identifier]["qa_checkpoint"]["completed"])
            self.assertEqual(rows[identifier]["qa_issues_summary"], "")
            self.assertEqual(rows[identifier]["qa_human_review_reason"], "")
        self.assertTrue(agent.closed)

    async def test_explicit_column_never_falls_back_across_row(self) -> None:
        write_jsonl(self.source, [{"id": "fallback", "指定列": MACHINE, "实际人标": HUMAN}])
        agent = CountingAgent()
        await qa.aprocess_batch(self.source, self.target, agent, column="指定列")
        self.assertEqual(agent.calls, [])
        self.assertEqual(read_jsonl(self.target)[0]["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")

    async def test_manual_column_with_only_machine_rows_skips_before_static_or_llm(self) -> None:
        write_jsonl(self.source, [{"id": "plain", "指定列": MACHINE}, {"id": "empty", "指定列": None}])
        agent = CountingAgent()
        with patch.object(qa, "run_static_rules", side_effect=AssertionError("skip must precede static rules")):
            self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent, column="指定列"), 2)
        self.assertEqual(agent.calls, [])
        self.assertTrue(all(row["qa_status"] == "SKIPPED_NOT_HUMAN_CAPTION" for row in read_jsonl(self.target)))

    async def test_autodetect_without_hits_fails_before_requests_and_output(self) -> None:
        write_jsonl(self.source, [{"id": "machine", "caption": MACHINE}])
        agent = CountingAgent()
        with self.assertRaisesRegex(ValueError, "人标|Caption|caption"):
            await qa.aprocess_batch(self.source, self.target, agent)
        self.assertEqual(agent.calls, [])
        self.assertFalse(self.target.exists())

    async def test_csv_filtering_preserves_leading_zeros_and_quoted_multiline_cells(self) -> None:
        self.source = self.source.with_suffix(".csv")
        originals = [{"id": "human", "video_id": "00013", "人工标注": HUMAN, "审核员": '张三, "复核"'},
                     {"id": "machine", "video_id": "00014", "人工标注": MACHINE, "审核员": "李四\n待复核"}]
        with self.source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(originals[0]))
            writer.writeheader()
            writer.writerows(originals)
        agent = CountingAgent()
        await qa.aprocess_batch(self.source, self.target, agent)
        self.assertEqual(agent.calls, [HUMAN])
        rows = {row["id"]: row for row in read_jsonl(self.target)}
        for original in originals:
            for key, value in original.items():
                self.assertEqual(rows[original["id"]][key], value)
        self.assertEqual(rows["machine"]["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")

    async def test_skipped_rows_are_completed_and_do_not_repeat_on_resume(self) -> None:
        write_jsonl(self.source, [{"id": "human", "人标": HUMAN}, {"id": "machine", "人标": MACHINE}])
        await qa.aprocess_batch(self.source, self.target, CountingAgent())
        first = self.target.read_bytes()
        resumed = CountingAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, resumed), 0)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(self.target.read_bytes(), first)

    async def test_reusing_checkpoint_with_different_selected_column_is_rejected(self) -> None:
        write_jsonl(self.source, [{"id": "human", "first": HUMAN, "second": OTHER_HUMAN}])
        await qa.aprocess_batch(self.source, self.target, CountingAgent(), column="first")
        changed = CountingAgent()
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, changed, column="second")
        self.assertEqual(changed.calls, [])

    async def test_review_and_readable_summary_include_duplicate_subject(self) -> None:
        duplicate = HUMAN.replace("<主体1>：白袍女子。", "<主体1>：白袍女子。\n<主体1>：黑衣男子。")
        write_jsonl(self.source, [{"id": 1, "审核员": "张三", "人标": duplicate}])
        await qa.aprocess_batch(self.source, self.target, CountingAgent())
        row = read_jsonl(self.target)[0]
        self.assertEqual(row["qa_status"], "NEED_HUMAN_REVIEW")
        self.assertIn("<主体1>", row["qa_missing_or_dup_subjects"])
        self.assertIn("主体", row["qa_issues_summary"])
        self.assertTrue(row["qa_human_review_reason"])
        self.assertTrue(row["qa_revised_caption"].startswith("## 任务类型"))

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl required for XLSX acceptance")
    async def test_xlsx_input_supports_multiline_caption_and_noncaption_cells(self) -> None:
        from openpyxl import Workbook
        self.source = self.source.with_suffix(".xlsx")
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["id", "video_id", "审核员", "人标", "机标caption"])
        sheet.append([1, "00017", "张三", HUMAN, MACHINE])
        sheet.append([2, "00018", "李四", "", MACHINE])
        workbook.save(self.source)
        workbook.close()
        agent = CountingAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent), 2)
        self.assertEqual(agent.calls, [HUMAN])
        rows = {row["id"]: row for row in read_jsonl(self.target)}
        self.assertEqual(rows[1]["video_id"], "00017")
        self.assertEqual(rows[1]["审核员"], "张三")
        self.assertEqual(rows[2]["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")


if __name__ == "__main__":
    unittest.main()
