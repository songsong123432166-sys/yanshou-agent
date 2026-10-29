"""Three-category automatic-fix policy and disabled findings."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import captionqa.caption_qa_agent as qa
from tests.test_caption_qa_upgrades import FakeSyncClient, GOOD, semantic_issue, semantic_reply


class AutoFixWhitelistTests(unittest.TestCase):
    def edit(self, old, new, kind, *, section="主体", auto=True):
        template = "<主体1>：身穿白袍。" if section == "主体" else "<主体1>奔跑。"
        text = GOOD.replace(template, old)
        issue = semantic_issue(section, old, "最小修复", kind)
        issue["can_auto_fix"] = auto
        client = FakeSyncClient(semantic_reply(issues=[issue], corrected_sections={section: new}))
        return text, qa.CaptionQAAgent(client=client, max_retries=0).check(text)

    def test_repeated_subject_prefix_is_format_fix_not_duplicate_definition(self):
        raw = GOOD.replace("<主体1>：", "<主体1>：<主体1>：<主体1>：")
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], raw)
        self.assertEqual(report["remaining_static_errors"], [])

    def test_true_repeated_definitions_remain_manual(self):
        raw = GOOD.replace("## 总结", "<主体1>：身穿黑袍。\n## 总结")
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_bare_definition_and_reference_do_not_cause_false_structure_alerts(self):
        for raw in (GOOD.replace("<主体1>：", "主体1："),
                    GOOD.replace("<主体1>奔跑", "主体1奔跑"),
                    GOOD.replace("<主体1>", "主体1")):
            with self.subTest(raw=raw):
                report = qa.CaptionQAAgent(dry_run=True).check(raw)
                self.assertEqual(report["status"], "AUTO_FIXED")
                self.assertEqual(report["revised_caption"], GOOD)

    def test_adjacent_redundancy_is_ignored(self):
        for old, new in (("暗红色红色长袍", "暗红色长袍"), ("她的的长袍", "她的长袍"),
                         ("慢慢慢地走", "慢慢地走")):
            with self.subTest(old=old):
                raw, report = self.edit(f"<主体1>：{old}。", f"<主体1>：{new}。", "衍字/叠词冗余")
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["revised_caption"], raw)

    def test_legitimate_reduplication_is_preserved_even_if_model_flags_it(self):
        for old, new in (("缓缓走动", "缓走动"), ("轻轻挥手", "轻挥手"), ("一动不动", "不动")):
            with self.subTest(old=old):
                raw, report = self.edit(f"<主体1>：{old}。", f"<主体1>：{new}。", "衍字/叠词冗余")
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["revised_caption"], raw)

    def test_known_typo_is_allowed(self):
        for wrong, correct in qa.UNAMBIGUOUS_TYPO_FIXES.items():
            with self.subTest(wrong=wrong):
                _, report = self.edit(f"<主体1>：{wrong}。", f"<主体1>：{correct}。", "无歧义错别字")
                self.assertEqual(report["status"], "AUTO_FIXED")

    def test_unknown_substitution_and_visual_fact_changes_are_rejected(self):
        for old, new in (("白色长袍", "红色长袍"), ("向左奔跑", "向右奔跑"),
                         ("站立", "奔跑"), ("白袍", "白衣")):
            with self.subTest(old=old):
                raw, report = self.edit(f"<主体1>：{old}。", f"<主体1>：{new}。", "无歧义错别字")
                self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
                self.assertEqual(report["revised_caption"], raw)

    def test_punctuation_only_repair_is_allowed(self):
        _, report = self.edit("<主体1>：眉头紧锁、身穿白袍。", "<主体1>：眉头紧锁，身穿白袍。", "标点符号硬伤")
        self.assertEqual(report["status"], "AUTO_FIXED")

    def test_punctuation_type_cannot_authorize_content_deletion_or_reordering(self):
        for old, new in (("身穿白袍，手持长剑", "身穿白袍"), ("向前奔跑", "奔跑向前"),
                         ("手持长剑", "手持长刀")):
            with self.subTest(old=old):
                raw, report = self.edit(f"<主体1>：{old}。", f"<主体1>：{new}。", "标点符号硬伤")
                self.assertEqual(report["revised_caption"], raw)
                self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")

    def test_broad_snippet_cannot_authorize_unique_detail_deletion(self):
        for kind in ("衍字/叠词冗余", "无歧义错别字", "错别字与衍字"):
            with self.subTest(kind=kind):
                raw, report = self.edit("<主体1>：身穿白袍，手持长剑，头戴金冠。", "<主体1>：身穿白袍。", kind)
                self.assertEqual(report["status"], "PASS" if kind in qa.DISABLED_LLM_ISSUE_TYPES else "NEED_HUMAN_REVIEW")
                self.assertEqual(report["revised_caption"], raw)

    def test_repeated_existing_entity_occurrence_cannot_be_deleted(self):
        raw, report = self.edit("<主体1>转身。<主体1>奔跑。", "<主体1>转身。", "衍字/叠词冗余", section="总结")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["revised_caption"], raw)

    def test_review_categories_cannot_be_auto_fixed_even_if_model_says_true(self):
        for kind in qa.REVIEW_LLM_ISSUE_TYPES:
            with self.subTest(kind=kind):
                raw, report = self.edit("<主体1>：身穿白袍。", "<主体1>：身穿红袍。", kind)
                self.assertEqual(report["revised_caption"], raw)
                self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
                self.assertFalse(report["llm_issues"][0]["can_auto_fix"])
                self.assertFalse(next(i for i in report["issues"] if i["source"] == "llm")["resolved"])

    def test_implicit_label_insertion_is_review_only(self):
        raw, report = self.edit("那名女子奔跑。", "<主体1>那名女子奔跑。", "隐式漏挂主体标签", section="总结")
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_whitelist_does_not_override_explicit_manual_issue(self):
        raw, report = self.edit("<主体1>：眼晴。", "<主体1>：眼睛。", "无歧义错别字", auto=False)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_other_fields_can_be_fixed_while_conflict_is_ignored(self):
        text = GOOD.replace("白袍", "白袍，露出眼晴").replace("<主体1>奔跑。", "<主体1>提起裙摆奔跑。")
        answer = semantic_reply(issues=[
            semantic_issue("主体", "眼晴", "眼睛", "无歧义错别字"),
            semantic_issue("总结", "裙摆", "与长袍冲突，需人工确认", "跨字段属性冲突"),
        ], corrected_sections={"主体": "<主体1>：身穿白袍，露出眼睛。", "总结": "<主体1>提起长袍下摆奔跑。"})
        report = qa.CaptionQAAgent(client=FakeSyncClient(answer), max_retries=0).check(text)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual(report["revised_caption"], text.replace("眼晴", "眼睛"))

    def test_mixed_manual_and_allowed_issues_keep_field_atomic(self):
        old = "<主体1>：身穿白袍，眼晴看着。"
        text = GOOD.replace("<主体1>：身穿白袍。", old)
        answer = semantic_reply(issues=[
            semantic_issue("主体", "眼晴", "眼睛", "无歧义错别字"),
            semantic_issue("主体", "看着", "成分残缺，需人工确认", "语法病句与语句不通顺"),
        ], corrected_sections={"主体": old.replace("眼晴", "眼睛")})
        report = qa.CaptionQAAgent(client=FakeSyncClient(answer), max_retries=0).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], text)

    def test_timestamp_change_is_not_a_punctuation_repair(self):
        raw = GOOD
        answer = semantic_reply(issues=[semantic_issue("自然指令", "0-4秒", "替换连接号", "标点符号硬伤")],
                                corrected_sections={"自然指令": "0—4秒，第0秒，<主体1>向前跑。"})
        report = qa.CaptionQAAgent(client=FakeSyncClient(answer), max_retries=0).check(raw)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_timestamp_protection_also_applies_outside_instruction(self):
        raw, report = self.edit("0-4秒，<主体1>奔跑。", "0—4秒，<主体1>奔跑。", "标点符号硬伤", section="总结")
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_punctuation_cannot_change_numeric_values(self):
        raw, report = self.edit("<主体1>：手持1,5米长的木棍。", "<主体1>：手持1.5米长的木棍。", "标点符号硬伤")
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_legacy_types_still_obey_edit_checks(self):
        _, report = self.edit("<主体1>：眼晴。", "<主体1>：眼睛。", "错别字与衍字")
        self.assertEqual(report["status"], "AUTO_FIXED")
        _, report = self.edit("<主体1>：眉头紧锁、身穿白袍。", "<主体1>：眉头紧锁，身穿白袍。", "标点与标签格式")
        self.assertEqual(report["status"], "AUTO_FIXED")

    def test_request_carries_exact_whitelist_and_typo_dictionary(self):
        client = FakeSyncClient(semantic_reply())
        qa.CaptionQAAgent(client=client).check(GOOD)
        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(set(payload["auto_fix_policy"]["allowed_issue_types"]), qa.AUTO_LLM_ISSUE_TYPES)
        self.assertEqual(payload["auto_fix_policy"]["unambiguous_typo_fixes"], qa.UNAMBIGUOUS_TYPO_FIXES)
        self.assertEqual(len(qa.AUTO_LLM_ISSUE_TYPES), 3)

    def test_completed_static_fixes_are_not_sent_as_pending_errors(self):
        raw = GOOD.replace("<主体1>：", "<主体1>：<主体1>：").replace("<主体1>奔跑。", "主体1奔跑")
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(raw)
        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["static_errors"], [])
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertTrue(report["static_errors"])
        self.assertTrue(all(issue["resolved"] for issue in report["static_errors"]))

    def test_unresolved_subject_structure_is_still_sent_to_model(self):
        raw = GOOD.replace("## 总结", "<主体2>：身穿红袍。\n## 总结")
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(raw)
        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertIn("missing_subjects", {i["issue_type"] for i in payload["static_errors"]})
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")

    def test_valid_question_and_exclamation_endings_are_preserved(self):
        for ending in ("？", "！", "?", "!"):
            with self.subTest(ending=ending):
                raw = GOOD.replace("<主体1>奔跑。", "<主体1>奔跑" + ending)
                self.assertEqual(qa.CaptionQAAgent(dry_run=True).check(raw)["revised_caption"], raw)

    def test_policy_change_invalidates_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "in.jsonl", Path(directory) / "out.jsonl"
            source.write_text(json.dumps({"caption": GOOD}) + "\n", encoding="utf-8")
            agent = qa.CaptionQAAgent(dry_run=True)
            qa.process_batch(source, target, agent)
            row = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(row["qa_checkpoint"]["auto_fix_policy"], qa.AUTO_FIX_POLICY_VERSION)
            del row["qa_checkpoint"]["auto_fix_policy"]
            target.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "来源/配置不匹配"):
                qa.process_batch(source, target, agent)


if __name__ == "__main__":
    unittest.main()
