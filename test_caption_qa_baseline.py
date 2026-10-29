"""Baseline regressions: declarations vs references, and no model dependency."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from test_caption_qa_upgrades import GOOD, FakeAsyncClient, semantic_reply


class BaselineTests(unittest.TestCase):
    def test_references_in_subject_descriptions_are_not_definitions(self):
        for description in (
            "剑斜背在<主体1>身后。", "音色同<主体1>。",
            "站在<主体1>的船上。", "剑背在主体1身后。", "看着<主体1>，靠近<主体1>。",
        ):
            with self.subTest(description=description):
                sections = {"主体": "<主体1>：男子。\n<主体2>：" + description,
                            "自然指令": "<主体1>看向<主体2>。"}
                self.assertEqual(qa.subject_coverage(sections)["defined_subjects"], ["<主体1>", "<主体2>"])
                self.assertFalse(any(i["issue_type"] == "duplicate_defined_subjects"
                                     for i in qa.run_static_rules(sections)))

    def test_real_repeated_declarations_are_still_reported(self):
        for text in ("<主体1>：男子。\n<主体1>：女子。", "<主体1>：男子。<主体1>：女子。"):
            with self.subTest(text=text):
                errors = qa.run_static_rules({"主体": text, "自然指令": "<主体1>走动。"})
                self.assertEqual([i["issue_type"] for i in errors], ["duplicate_defined_subjects"])

    def test_disabled_duplicate_prefix_is_not_changed_or_counted_twice(self):
        sections = {"主体": "<主体1>：<主体1>：男子。", "自然指令": "<主体1>走动。"}
        self.assertEqual(qa.run_static_rules(sections), [])
        self.assertEqual(qa.apply_static_fixes(sections), sections)

    def test_reference_alone_does_not_declare_a_subject(self):
        sections = {"主体": "<主体1>：音色同<主体2>。", "自然指令": "<主体1>看向<主体2>。"}
        self.assertEqual(qa.subject_coverage(sections)["defined_subjects"], ["<主体1>"])
        self.assertEqual(qa.subject_coverage(sections)["ghost_subjects"], ["<主体2>"])

    def test_existing_tag_spaces_are_fixed_without_changing_numbers_or_outer_space(self):
        original = {"主体": "  < 主体1 >：男子。  \r\n",
                    "自然指令": "<主体 1>看向<场景3 >。\n"}
        fixed = {"主体": "  <主体1>：男子。  \r\n", "自然指令": "<主体1>看向<场景3>。\n"}
        self.assertEqual(qa.apply_static_fixes(original), fixed)
        self.assertEqual(len(qa.run_static_rules(original)), 3)
        self.assertEqual(qa.run_static_rules(fixed), [])
        self.assertEqual(qa.apply_static_fixes(fixed), fixed)

    def test_baseline_leaves_suspected_typos_and_grammar_for_human_review(self):
        raw = GOOD.replace("身穿白袍。", "身穿白袍，带着帽子，手手抬起。")
        with patch.object(qa, "run_llm_semantic_qa", side_effect=AssertionError("model forbidden")):
            reports = [qa.CaptionQAAgent(dry_run=True).check(raw) for _ in range(2)]
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0]["status"], "PASS")
        self.assertFalse(reports[0]["semantic_qa_performed"])
        self.assertEqual(reports[0]["revised_caption"], raw)

    def test_old_model_job_rechecks_as_baseline_without_key_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = web.App(root / "Downloads", root / "work")
            restored = None
            try:
                data = (json.dumps({"id": "1", "caption": GOOD}) + "\n").encode()
                fake = FakeAsyncClient(semantic_reply())
                real_agent = qa.CaptionQAAgent
                with patch.object(web, "read_saved_key", return_value="test-key"), patch.object(
                    qa, "CaptionQAAgent", side_effect=lambda **kw: real_agent(async_client=fake, **kw)
                ):
                    item = app.create_job("business.jsonl", io.BytesIO(data), len(data), review_first=False, dry_run=False)
                    app.executor.submit(lambda: None).result(timeout=10)
                job = app.jobs[item["id"]]
                self.assertEqual(job.status, "done", job.error)
                self.assertEqual(len(fake.calls), 1)
                app.save_history()
                old_cache, old_output = job.cache, job.output
                row = json.loads(old_cache.read_text())
                row["qa_checkpoint"]["auto_fix_policy"] = qa.AUTO_FIX_POLICY_VERSION - 1
                old_cache.write_text(json.dumps(row) + "\n")
                original = old_cache.read_bytes(), old_output.read_bytes()
                with patch.object(web, "read_saved_key", side_effect=AssertionError("key forbidden")):
                    restored = web.App(app.output_dir, app.work_dir)
                    restored.retry(job.id)
                    restored.executor.submit(lambda: None).result(timeout=10)
                result = restored.jobs[job.id]
                self.assertTrue(result.dry_run)
                self.assertEqual(result.status, "done", result.error)
                self.assertNotEqual(result.cache, old_cache)
                self.assertNotEqual(result.output, old_output)
                self.assertEqual((old_cache.read_bytes(), old_output.read_bytes()), original)
                checked = json.loads(result.cache.read_text())
                self.assertEqual(checked["qa_scope"], "static_only")
                self.assertFalse(checked["qa_report"]["semantic_qa_performed"])
            finally:
                if restored:
                    restored.close()
                app.close()


if __name__ == "__main__":
    unittest.main()
