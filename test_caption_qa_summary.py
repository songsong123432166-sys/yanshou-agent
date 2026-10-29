"""Per-check counts must use Caption rows and the latest checkpoint version."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_summary import ISSUE_CHECK, build_check_report
from test_caption_qa_sniff import CountingAgent, write_jsonl
from test_caption_qa_upgrades import GOOD


def issue(kind, resolved=False, source="llm"):
    return {"issue_type": kind, "resolved": resolved, "source": source}


def row(index, *issues, semantic=True, review=False, status="NEED_HUMAN_REVIEW", scope="static_and_semantic"):
    return {"row_index": index, "id": "same-id", "qa_status": status, "qa_scope": scope,
            "qa_checkpoint": {"completed": True},
            "qa_report": {"issues": list(issues), "semantic_qa_performed": semantic,
                          "need_human_review": review, "qa_scope": scope}}


def items(report):
    return {item["key"]: item for item in report["items"]}


class CheckSummaryTests(unittest.TestCase):
    def test_same_check_deduplicates_occurrences_and_partitions_fixed_and_review(self):
        report = build_check_report([
            row(1, issue("missing_punctuation", True, "static"), issue("标点符号硬伤", True),
                issue("标点符号硬伤")),
            row(2, issue("missing_punctuation", True, "static"), issue("标点符号硬伤", True))])
        punctuation = items(report)["punctuation"]
        self.assertEqual((punctuation["affected"], punctuation["fixed"], punctuation["review"]), (2, 1, 1))
        self.assertEqual(report["checked"], 2)

    def test_latest_retry_replaces_previous_check_results(self):
        report = build_check_report([row(1, issue("无歧义错别字")), row(1, status="PASS")])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(items(report)["typos"]["affected"], 0)

    def test_rows_are_independent_of_ids_and_multiple_checks_can_overlap(self):
        first = row(1, issue("bare_tags", True, "static"), issue("missing_punctuation", True, "static"))
        second = row(2, issue("bare_tags", True, "static"))
        first["id"] = ""
        report = build_check_report([first, second])
        self.assertEqual(items(report)["bare_tags"]["affected"], 2)
        self.assertEqual(sum(i["affected"] for i in report["items"]), 3)

    def test_skips_and_input_failures_are_excluded(self):
        report = build_check_report([
            row(1, issue("input_error"), scope="input_error"),
            row(2, status="SKIPPED_NOT_HUMAN_CAPTION"), row(3, status="PASS")])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(sum(i["affected"] for i in report["items"]), 0)

    def test_disabled_findings_do_not_reappear_and_guards_are_separate(self):
        report = build_check_report([
            row(1, issue("跨字段属性冲突"), issue("重复标签前缀"), issue("衍字/叠词冗余"), review=True),
            row(2, issue("unsafe_llm_edit", source="merge"), review=True),
            row(3, review=True)])
        self.assertEqual(sum(i["affected"] for i in report["items"]), 0)
        self.assertEqual(report["correction_review_rows"], 1)
        self.assertEqual(report["other_review_rows"], 1)

    def test_taxonomy_and_no_semantic_checks_are_explicit(self):
        report = build_check_report([row(1, semantic=False)])
        self.assertEqual(report["semantic_checked"], 0)
        self.assertEqual(len(report["items"]), 10)
        self.assertFalse(items(report)["symbols"]["requires_semantic"])
        self.assertTrue(items(report)["typos"]["requires_semantic"])
        self.assertFalse(items(report)["missing_subjects"]["requires_semantic"])
        for kind, key in ISSUE_CHECK.items():
            with self.subTest(kind=kind):
                self.assertEqual(items(build_check_report([row(1, issue(kind))]))[key]["review"], 1)

    def test_job_refresh_accumulates_latest_rows_and_waits_for_complete_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/"cache.jsonl"
            first = row(1, issue("missing_punctuation", True, "static"))
            retry = row(1, issue("标点符号硬伤"))
            cache.write_text(json.dumps(first)+"\n"+json.dumps(retry))
            job = web.Job("a"*32, "test.csv", root/"input.csv", root/"ids.txt", cache, False, None, None, 8)
            self.assertEqual(items(job.snapshot()["check_report"])["punctuation"]["fixed"], 1)
            with cache.open("a") as stream:
                stream.write("\n")
            report = job.snapshot()["check_report"]
            self.assertEqual(report["checked"], 1)
            self.assertEqual(items(report)["punctuation"]["fixed"], 0)
            self.assertEqual(items(report)["punctuation"]["review"], 1)
            self.assertEqual(job.snapshot()["check_report"], report)

    def test_future_id_exports_include_report_without_changing_id_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/"input.jsonl", root/"ids.txt"
            write_jsonl(source, [{"id": "0001", "caption": GOOD.replace("身穿白袍。", "身穿白袍")},
                                 {"id": "0002", "caption": GOOD}])
            asyncio.run(qa.aprocess_batch(source, output, CountingAgent()))
            self.assertEqual(output.read_text(), "0001\n")
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(items(summary["check_report"])["punctuation"]["fixed"], 1)
            self.assertIn("标点问题：涉及 1 条，已修复 1 条，需复核 0 条。", output.with_name("ids_检测说明.txt").read_text())

    def test_old_completed_history_gets_report_without_model_or_keychain_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = web.App(root/"Downloads", root/"work")
            restored = None
            try:
                import io
                data = (json.dumps({"id": "0001", "caption": GOOD})+"\n").encode()
                item = app.create_job("test.jsonl", io.BytesIO(data), len(data), review_first=False, dry_run=True)
                app.executor.submit(lambda: None).result(timeout=10)
                original = app.jobs[item["id"]].cache.read_bytes()
                with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
                    restored = web.App(app.output_dir, app.work_dir)
                    report = restored.jobs[item["id"]].snapshot()["check_report"]
                self.assertEqual(report["checked"], 1)
                self.assertEqual(app.jobs[item["id"]].cache.read_bytes(), original)
            finally:
                if restored:
                    restored.close()
                app.close()


if __name__ == "__main__":
    unittest.main()
