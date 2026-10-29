"""User policy: only four fields are exempt; dialogue needs outer punctuation."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from test_caption_qa_sniff import CountingAgent, write_jsonl
from test_caption_qa_upgrades import GOOD, FakeSyncClient, semantic_issue, semantic_reply


EXAMPLE = "0-5秒，在<场景1>中，手持镜头Lazy Susan运镜从身前到身侧，中景拍摄<主体1>位于画面右侧，伸左臂做出自拍姿势，一边说道{何が原因かわからない}，一边转头看向其右侧。第 1 秒，Lazy Susan运镜从身侧回到身前之后稍稍回摇，中景拍摄<主体1>头转回面对镜头，期间<主体1>说道{私は異世界に来たようだ。}"


class PunctuationPolicyTests(unittest.TestCase):
    def test_named_metadata_fields_and_known_equivalent_export_fields_are_exempt(self):
        expected = {"任务类型", "similarity_label", "video_similarity_tag", "audio_similarity_tag",
                    "task_type", "subtask_type", "similarity_tag", "视听标签"}
        self.assertEqual(set(qa.PUNCTUATION_EXEMPT_SECTIONS), expected)
        sections = {name: "  主体9|视频1  \r\n" for name in expected}
        self.assertEqual(qa.run_static_rules(sections), [])
        self.assertEqual(qa.apply_static_fixes(sections), sections)

    def test_every_other_field_requires_terminal_punctuation(self):
        for name in ("提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令", "画面元素", "自定义字段", "other_label"):
            with self.subTest(name=name):
                errors = qa.run_static_rules({name: "普通内容"})
                self.assertEqual([e["issue_type"] for e in errors], ["missing_punctuation"])
                self.assertEqual(qa.apply_static_fixes({name: "普通内容"})[name], "普通内容。")

    def test_none_value_in_nonexempt_field_also_requires_period(self):
        self.assertEqual(qa.apply_static_fixes({"画面元素": "无\n"}), {"画面元素": "无。\n"})

    def test_user_lazy_susan_example_gets_exactly_one_outer_period(self):
        raw = GOOD.replace("0-4秒，第0秒，<主体1>向前跑。", EXAMPLE)
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual(report["revised_caption"], raw.replace(EXAMPLE, EXAMPLE + "。"))
        self.assertEqual(report["remaining_static_errors"], [])
        self.assertEqual(qa.TIMESTAMP_RE.findall(raw), qa.TIMESTAMP_RE.findall(report["revised_caption"]))
        self.assertEqual(qa.TAG_RE.findall(raw), qa.TAG_RE.findall(report["revised_caption"]))

    def test_inner_punctuation_and_closing_symbols_do_not_replace_outer_ending(self):
        for text in ('说道{你好。}', '说道{你好？}', '说道“你好。”', '说道「你好。」', '描述（内容。）'):
            with self.subTest(text=text):
                fixed = qa.apply_static_fixes({"自然指令": text})
                self.assertEqual(fixed["自然指令"], text + "。")
                self.assertEqual(qa.apply_static_fixes(fixed), fixed)

    def test_valid_existing_terminal_punctuation_is_not_duplicated(self):
        for text in ("内容。", "你好？", "注意！", "Hello?", "Stop!", '说道{你好。}。'):
            with self.subTest(text=text):
                self.assertEqual(qa.apply_static_fixes({"提示": text})["提示"], text)
                self.assertEqual(qa.run_static_rules({"提示": text}), [])

    def test_midparagraph_braces_with_following_text_do_not_get_a_period(self):
        for text in ('说道{你好}，然后转身。', '说道{你好。}之后转身。',
                     '说道{你好}并指向{出口}，然后离开。'):
            with self.subTest(text=text):
                sections = {"自然指令": text}
                self.assertEqual(qa.run_static_rules(sections), [])
                self.assertEqual(qa.apply_static_fixes(sections), sections)

    def test_only_paragraph_end_gets_period_even_with_multiple_braces(self):
        text = "说道{你好}，然后说{再见}  \r\n说道{回来}，转身离开\r\n"
        expected = "说道{你好}，然后说{再见}。  \r\n说道{回来}，转身离开。\r\n"
        self.assertEqual(qa.apply_static_fixes({"自然指令": text})["自然指令"], expected)
        self.assertEqual(len(qa.run_static_rules({"自然指令": text})), 2)

    def test_llm_cannot_add_period_after_midparagraph_brace(self):
        original = "<主体1>说道{你好}，然后转身。"
        for candidate in ("<主体1>说道{你好}。，然后转身。", "<主体1>说道{你好}。然后转身。"):
            with self.subTest(candidate=candidate):
                text = GOOD.replace("0-4秒，第0秒，<主体1>向前跑。", original)
                client = FakeSyncClient(semantic_reply(
                    issues=[semantic_issue("自然指令", original, "给大括号后补句号", "标点符号硬伤")],
                    corrected_sections={"自然指令": candidate}))
                report = qa.CaptionQAAgent(client=client, max_retries=0).check(text)
                self.assertEqual(report["revised_caption"], text)
                self.assertTrue(any(i["issue_type"] == "unsafe_llm_edit" for i in report["issues"]))
                policy = json.loads(client.calls[0]["messages"][1]["content"])["auto_fix_policy"]
                self.assertEqual(policy["outer_sentence_ending_scope"], "paragraph_end_only")
                self.assertFalse(policy["add_period_after_midparagraph_brace"])

    def test_whitespace_crlf_and_blank_lines_are_preserved(self):
        raw = "第一行  \t\r\n\r\n第二行\u3000\r\n"
        self.assertEqual(qa.apply_static_fixes({"基调": raw})["基调"], "第一行。  \t\r\n\r\n第二行。\u3000\r\n")
        self.assertEqual(qa.apply_static_fixes({"画面元素": " \r\n\t"})["画面元素"], " \r\n\t")

    def test_frame_elements_only_receive_punctuation_not_tag_rewrites(self):
        original = "主体3 <场景1>：<场景1>：原样"
        self.assertEqual(qa.apply_static_fixes({"画面元素": original})["画面元素"], original + "。")
        self.assertEqual([i["issue_type"] for i in qa.run_static_rules({"画面元素": original})], ["missing_punctuation"])

    def test_llm_cannot_remove_the_required_outer_period(self):
        text = GOOD.replace("0-4秒，第0秒，<主体1>向前跑。", "<主体1>说道{你好。}。")
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("自然指令", "。}。", "删除外层句号", "标点符号硬伤")],
            corrected_sections={"自然指令": "<主体1>说道{你好。}"}))
        report = qa.CaptionQAAgent(client=client, max_retries=0).check(text)
        self.assertEqual(report["revised_caption"], text)
        self.assertTrue(any(i["issue_type"] == "unsafe_llm_edit" for i in report["issues"]))
        policy = json.loads(client.calls[0]["messages"][1]["content"])["auto_fix_policy"]
        self.assertTrue(policy["require_outer_sentence_ending"])
        self.assertEqual(set(policy["punctuation_exempt_sections"]), set(qa.PUNCTUATION_EXEMPT_SECTIONS))

    def test_problem_id_list_catches_newly_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.jsonl", Path(directory) / "ids.txt"
            raw = GOOD + "## 提示\n参考素材\n## 基调\n写实\n## 画面元素\n无\n"
            write_jsonl(source, [{"id": "0001", "caption": raw}, {"id": "0002", "caption": GOOD}])
            asyncio.run(qa.aprocess_batch(source, output, CountingAgent()))
            self.assertEqual(output.read_text(), "0001\n")

    def test_old_policy_cache_is_rejected_without_model_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.jsonl", Path(directory) / "ids.txt"
            write_jsonl(source, [{"id": "0001", "caption": GOOD}])
            asyncio.run(qa.aprocess_batch(source, output, CountingAgent()))
            cache = output.with_suffix(".jsonl")
            row = json.loads(cache.read_text())
            row["qa_checkpoint"]["auto_fix_policy"] = qa.AUTO_FIX_POLICY_VERSION - 1
            cache.write_text(json.dumps(row) + "\n")
            agent = CountingAgent()
            with self.assertRaisesRegex(ValueError, "不匹配"):
                asyncio.run(qa.aprocess_batch(source, output, agent))
            self.assertEqual(agent.calls, [])

    def test_historical_results_remain_readable_and_recheck_uses_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = web.App(root / "Downloads", root / "work")
            restored = None
            try:
                data = (json.dumps({"id": "0001", "caption": GOOD}) + "\n").encode()
                item = app.create_job("business.jsonl", io.BytesIO(data), len(data), review_first=False, dry_run=True)
                app.executor.submit(lambda: None).result(timeout=10)
                job = app.jobs[item["id"]]
                app.save_history()
                old_cache, old_output = job.cache, job.output
                row = json.loads(old_cache.read_text())
                row["qa_checkpoint"]["auto_fix_policy"] = qa.AUTO_FIX_POLICY_VERSION - 1
                old_cache.write_text(json.dumps(row) + "\n")
                previous_bytes = old_cache.read_bytes()
                resolution = {"source_sha256": row["qa_checkpoint"]["source_sha256"], "rows": {}}
                old_cache.with_suffix(".id_resolutions.json").write_text(json.dumps(resolution))
                with patch.object(web, "read_saved_key", side_effect=AssertionError("no API")):
                    restored = web.App(app.output_dir, app.work_dir)
                    job = restored.jobs[item["id"]]
                    snapshot = job.snapshot()
                    self.assertEqual(snapshot["status"], "done")
                    self.assertTrue(snapshot["output_ready"])
                    self.assertTrue(snapshot["outdated_policy"])
                    self.assertFalse(snapshot["stale_results"])
                    self.assertIn("旧规则", snapshot["policy_note"])
                    restored.retry(job.id)
                    restored.executor.submit(lambda: None).result(timeout=10)
                self.assertEqual(job.status, "done", job.error)
                self.assertFalse(job.outdated_policy)
                self.assertNotEqual(job.cache, old_cache)
                self.assertNotEqual(job.output, old_output)
                self.assertEqual(old_cache.read_bytes(), previous_bytes)
                self.assertTrue(old_output.exists())
                self.assertEqual(json.loads(job.cache.with_suffix(".id_resolutions.json").read_text()), resolution)
            finally:
                if restored:
                    restored.close()
                app.close()


if __name__ == "__main__":
    unittest.main()
