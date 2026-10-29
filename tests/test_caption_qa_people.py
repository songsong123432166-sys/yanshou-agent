"""Annotator attribution and rate ranking use existing Caption QA only."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import captionqa.caption_qa_agent as qa
import captionqa.caption_qa_web as web
from captionqa.caption_qa_people import PeopleTracker, build_people_report, export_people_report
from tests.test_caption_qa_sniff import CountingAgent, write_jsonl
from tests.test_caption_qa_summary import issue, row
from tests.test_caption_qa_upgrades import GOOD, FakeAsyncClient, semantic_reply


def record(index, person, *, problem=False, fixed=False, completed=True, column="标注员"):
    issues = [issue("bare_tags", True, "static")] if fixed else [issue("语法病句与语句不通顺")] if problem else []
    value = row(index, *issues, review=problem and not fixed,
                status="AUTO_FIXED" if fixed else "NEED_HUMAN_REVIEW" if problem else "PASS")
    value[column] = person
    value["qa_checkpoint"]["completed"] = completed
    if not completed:
        value["qa_report"]["semantic_qa_performed"] = False
    return value


class PeopleReportTests(unittest.TestCase):
    def test_rate_sort_and_ties_include_denominators(self):
        records = [record(1, "甲", problem=True), record(2, "甲"),
                   record(3, "乙", problem=True), record(4, "丙", fixed=True), record(5, "丙", fixed=True)]
        people = build_people_report(records)["people"]
        self.assertEqual([p["name"] for p in people], ["丙", "乙", "甲"])
        self.assertEqual([p["rank"] for p in people], [1, 1, 3])
        self.assertEqual([p["problem_rate"] for p in people], [100.0, 100.0, 50.0])
        self.assertEqual(people[0]["fixed_rows"], 2)
        self.assertEqual(people[1]["review_rows"], 1)

    def test_second_header_detects_annotator_and_excludes_header_skip(self):
        header = row(1, status="SKIPPED_NOT_HUMAN_CAPTION")
        header.update({"__column_18": "标注员", "音频质检管理": "音频标注人", "qa_skip_reason": "table_header"})
        report = build_people_report([record(2, "甲", problem=True, column="__column_18"), header])
        self.assertTrue(report["available"])
        self.assertEqual(report["column"], "__column_18")
        self.assertEqual(report["people"][0]["records"], 1)

    def test_audio_and_review_columns_cannot_be_used_as_annotator(self):
        for label in ("音频标注人", "音频标注员", "审核人", "质检人", "reviewer"):
            with self.subTest(label=label):
                self.assertFalse(build_people_report([record(1, "甲", column=label)])["available"])

    def test_ambiguous_annotator_columns_do_not_choose_someone(self):
        value = record(1, "甲")
        value["标注人"] = "乙"
        self.assertFalse(build_people_report([value])["available"])

    def test_duplicate_annotator_headers_are_also_ambiguous(self):
        value = record(1, "甲")
        value["标注员__2"] = "乙"
        self.assertFalse(build_people_report([value])["available"])

    def test_multi_author_and_blank_records_are_not_charged_to_individuals(self):
        report = build_people_report([record(1, "甲\n乙", problem=True), record(2, " "), record(3, "甲")])
        self.assertEqual(report["unattributed_records"], 2)
        self.assertEqual(report["unattributed_problem_rows"], 1)
        self.assertEqual(report["people"][0]["checked"], 1)
        self.assertEqual(report["people"][0]["problem_rows"], 0)

    def test_only_valid_trailing_date_is_normalized(self):
        report = build_people_report([record(1, "甲 10/22"), record(2, "甲（10/23）"), record(3, "甲"),
                                      record(4, "甲 99/99")])
        by_name = {p["name"]: p for p in report["people"]}
        self.assertEqual(by_name["甲"]["checked"], 3)
        self.assertEqual(report["unattributed_records"], 1)

    def test_user_confirmed_compact_dates_and_plus_suffixes_are_merged(self):
        labels = ["甲", "甲1020", "甲 1020", "甲+", "甲＋", "甲10/22+", "甲+（10/23）"]
        report = build_people_report(record(i, name) for i, name in enumerate(labels, 1))
        self.assertEqual(len(report["people"]), 1)
        self.assertEqual(report["people"][0]["name"], "甲")
        self.assertEqual(report["people"][0]["checked"], len(labels))
        self.assertEqual(report["people"][0]["source_names"], labels)

    def test_incomplete_rows_do_not_inflate_rate_denominator(self):
        report = build_people_report([record(1, "甲", problem=True), record(2, "甲", completed=False),
                                      record(3, "乙", completed=False)])
        first, second = report["people"]
        self.assertEqual(first["problem_rate"], 100.0)
        self.assertEqual(first["records"], 2)
        self.assertEqual(first["not_checked"], 1)
        self.assertIsNone(second["rank"])
        self.assertIsNone(second["problem_rate"])

    def test_latest_checkpoint_replaces_previous_person_and_result(self):
        report = build_people_report([record(1, "甲", problem=True), record(1, "乙")])
        self.assertEqual([p["name"] for p in report["people"]], ["乙"])
        self.assertEqual(report["people"][0]["problem_rows"], 0)

    def test_tracker_does_not_keep_caption_body(self):
        value = record(1, "甲")
        value["caption"] = GOOD
        tracker = PeopleTracker()
        tracker.add(value)
        self.assertNotIn("caption", tracker.rows[1]["metadata"])

    def test_export_contains_counts_and_no_caption_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_people_report([record(1, "甲", problem=True)])
            path = export_people_report(report, Path(directory)/"ids.txt")
            text = Path(path).read_text()
            self.assertIn("甲：已检 1 条，涉及问题 1 条", text)
            self.assertIn("病句 / 语句不通顺 1 条", text)
            self.assertNotIn("身穿白袍", text)

    def test_future_exports_generate_separate_person_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/"input.jsonl", root/"ids.txt"
            write_jsonl(source, [{"id": "0001", "标注员": "甲", "caption": GOOD.replace("身穿白袍。", "身穿白袍")}])
            asyncio.run(qa.aprocess_batch(source, output, CountingAgent()))
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertTrue(Path(summary["people_report_path"]).is_file())
            self.assertEqual(output.read_text(), "0001\n")

    def test_names_are_not_sent_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/"input.jsonl", root/"ids.txt"
            write_jsonl(source, [{"id": "0001", "标注员": "私有标注员姓名", "caption": GOOD}])
            client = FakeAsyncClient(semantic_reply())
            asyncio.run(qa.aprocess_batch(source, output, qa.CaptionQAAgent(async_client=client, max_retries=0)))
            self.assertNotIn("私有标注员姓名", json.dumps(client.calls, ensure_ascii=False))

    def test_restore_builds_people_report_without_model_or_keychain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = web.App(root/"Downloads", root/"work")
            restored = None
            try:
                data = (json.dumps({"id": "1", "标注员": "甲", "caption": GOOD})+"\n").encode()
                item = app.create_job("people.jsonl", io.BytesIO(data), len(data), review_first=False, dry_run=True)
                app.executor.submit(lambda: None).result(timeout=10)
                with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
                    restored = web.App(app.output_dir, app.work_dir)
                    snapshot = restored.jobs[item["id"]].snapshot()
                self.assertEqual(snapshot["people_report"]["people"][0]["name"], "甲")
                self.assertTrue(snapshot["people_report_path"])
            finally:
                if restored:
                    restored.close()
                app.close()


if __name__ == "__main__":
    unittest.main()
