"""Disabled checks must never change text or contribute problem IDs."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import captionqa.caption_qa_agent as qa
from tests.test_caption_qa_sniff import write_jsonl
from tests.test_caption_qa_upgrades import GOOD, FakeAsyncClient, FakeSyncClient, semantic_issue, semantic_reply


class DisabledCategoryTests(unittest.TestCase):
    def check(self, raw=GOOD, **changes):
        return qa.CaptionQAAgent(client=FakeSyncClient(semantic_reply(**changes)), max_retries=0).check(raw)

    def test_repeated_prefix_is_neither_reported_nor_modified_or_called_duplicate_definition(self):
        for prefix in ("<主体1>：<主体1>：", "<主体1><主体1><主体1>："):
            raw = GOOD.replace("<主体1>：", prefix)
            report = qa.CaptionQAAgent(dry_run=True).check(raw)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["issues"], [])
            self.assertEqual(report["revised_caption"], raw)

    def test_disabled_model_findings_and_generic_review_reason_are_ignored(self):
        for kind in ("重复标签前缀", "衍字/叠词冗余", "跨字段属性冲突", "属性冲突"):
            report = self.check(issues=[semantic_issue("主体", "白袍", "最小修复", kind)],
                need_human_review=True, human_review_reason="需要人工复核",
                corrected_sections={"主体": "<主体1>：身穿红袍。"})
            self.assertEqual(report["status"], "PASS", kind)
            self.assertEqual(report["issues"], [])
            self.assertEqual(report["revised_caption"], GOOD)
            self.assertFalse(report["need_human_review"])

    def test_reason_only_disabled_findings_do_not_trigger_review(self):
        for reason in ("长袍与裙摆不一致", "跨字段属性冲突", "重复标签前缀需核对", "衍字/叠词冗余"):
            report = self.check(need_human_review=True, human_review_reason=reason)
            self.assertEqual(report["status"], "PASS", reason)
            self.assertEqual(report["human_review_reason"], "")

    def test_disabled_finding_cannot_be_relabelled_as_grammar(self):
        report = self.check(issues=[semantic_issue("主体", "白袍", "服饰属性冲突，改为裙子", "语法病句与语句不通顺")])
        self.assertEqual(report["status"], "PASS")

    def test_reason_only_conflict_corrected_text_is_ignored(self):
        report = self.check(need_human_review=True, human_review_reason="服饰属性冲突",
            corrected_sections={"主体": "<主体1>：身穿红袍。"})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], GOOD)

    def test_legacy_format_type_cannot_reenable_prefix_cleanup(self):
        raw = GOOD.replace("<主体1>：", "<主体1>：<主体1>：")
        report = self.check(raw, issues=[semantic_issue("主体", "<主体1>：<主体1>：", "最小修复", "标点与标签格式")],
            corrected_sections={"主体": "<主体1>：身穿白袍。"})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], raw)

    def test_legacy_or_typo_label_cannot_enable_redundant_deletion(self):
        raw = GOOD.replace("白袍", "暗红色红色白袍")
        for kind in ("错别字与衍字", "无歧义错别字", "语法病句与语句不通顺"):
            report = self.check(raw, issues=[semantic_issue("主体", "暗红色红色", "最小修复", kind)],
                corrected_sections={"主体": "<主体1>：身穿暗红色白袍。"})
            self.assertEqual(report["status"], "PASS", kind)
            self.assertEqual(report["revised_caption"], raw)

    def test_typo_in_same_field_still_fixes_while_conflict_edit_is_removed(self):
        raw = GOOD.replace("白袍", "白袍，露出眼晴")
        report = self.check(raw, issues=[
            semantic_issue("主体", "眼晴", "眼睛", "无歧义错别字"),
            semantic_issue("主体", "白袍", "服饰属性冲突", "跨字段属性冲突")],
            need_human_review=True, human_review_reason="服饰属性冲突，需要人工复核",
            corrected_sections={"主体": "<主体1>：身穿红袍，露出眼睛。"})
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual(report["revised_caption"], raw.replace("眼晴", "眼睛"))
        self.assertEqual([i["issue_type"] for i in report["issues"]], ["无歧义错别字"])

    def test_enabled_grammar_reason_survives_disabled_findings(self):
        report = self.check(issues=[semantic_issue("主体", "白袍", "属性冲突", "跨字段属性冲突")],
            need_human_review=True, human_review_reason="属性冲突；自然指令存在语序颠倒，需人工核对")
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertIn("语序颠倒", report["human_review_reason"])
        self.assertNotIn("属性冲突", report["human_review_reason"])

    def test_bare_tags_and_periods_still_fix_without_collapsing_prefix(self):
        raw = GOOD.replace("<主体1>：身穿白袍。", "主体1：主体1：身穿白袍")
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("<主体1>：<主体1>：身穿白袍。", report["revised_caption"])
        self.assertEqual({i["issue_type"] for i in report["issues"]}, {"bare_tags", "missing_punctuation"})

    def test_disabled_checks_do_not_hide_missing_instruction_subjects(self):
        raw = GOOD.replace("## 总结", "<主体2>：红衣女子。\n## 总结")
        report = self.check(raw, issues=[semantic_issue("主体", "白袍", "属性冲突", "跨字段属性冲突")])
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["subject_check"]["missing_in_instruction"], ["<主体2>"])

    def test_redundant_punctuation_remains_an_enabled_check(self):
        raw = GOOD.replace("白袍", "白袍、，露出眼睛")
        report = self.check(raw, issues=[semantic_issue("主体", "、，", "删除冗余顿号", "标点符号硬伤")],
            corrected_sections={"主体": "<主体1>：身穿白袍，露出眼睛。"})
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual(report["revised_caption"], raw.replace("、，", "，"))

    def test_disabled_only_sample_not_exported_to_problem_id_list(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"input.jsonl", Path(directory)/"ids.txt"
            write_jsonl(source, [{"id": "0001", "caption": GOOD}])
            answer = semantic_reply(issues=[semantic_issue("主体", "白袍", "属性冲突", "跨字段属性冲突")],
                need_human_review=True, human_review_reason="长袍与裙摆属性冲突")
            agent = qa.CaptionQAAgent(async_client=FakeAsyncClient(answer), max_retries=0)
            asyncio.run(qa.aprocess_batch(source, output, agent))
            self.assertEqual(output.read_text(), "")
            self.assertTrue(json.loads(output.with_suffix(".summary.json").read_text())["complete"])
            self.assertNotIn("属性冲突", output.with_name("ids_检测说明.txt").read_text())


if __name__ == "__main__":
    unittest.main()
