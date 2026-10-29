"""Defined subjects must be referenced in the natural instruction itself."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

import captionqa.caption_qa_agent as qa
from tests.test_caption_qa_sniff import CountingAgent, read_jsonl, write_jsonl
from tests.test_caption_qa_upgrades import FakeSyncClient, FakeAsyncClient, semantic_issue, semantic_reply


CAPTION = (
    "## 主体\n<主体1>：白衣女子。\n<主体2>：黑衣男子。\n<主体3>：红衣女子。\n"
    "## 总结\n<主体1>、<主体2>和<主体3>站在一起。\n"
    "## 自然指令\n0-5秒，<主体1>挥手。\n"
)


class InstructionSubjectTests(unittest.TestCase):
    def test_english_instruction_alias_reads_subjects_and_preserves_original_heading(self):
        complete=CAPTION.replace('<主体1>挥手。','<主体1>、<主体2>和<主体3>挥手。')
        for heading in ('## natural_instruction\n','##natural_instruction\n',
                        'natural_instruction\n','## natural_instruction：',
                        'natural_instruction    '):
            with self.subTest(heading=heading):
                text=complete.replace('## 自然指令\n',heading)
                report=qa.CaptionQAAgent(dry_run=True).check(text)
                self.assertEqual(report['subject_check']['missing_in_instruction'],[])
                self.assertEqual(report['revised_caption'],text)

    def test_english_alias_keeps_real_missing_subjects_and_yields_to_current_name(self):
        text=CAPTION.replace('## 自然指令','## natural_instruction')
        report=qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report['subject_check']['missing_subjects'],['<主体2>','<主体3>'])
        selected=qa.parse_caption_sections(text+'## 自然指令\n<主体2>挥手。')
        self.assertEqual(selected['自然指令'],'<主体2>挥手。')
        self.assertTrue(selected.field_resolution_notes)

    def test_english_instruction_is_sent_to_model_and_rebuilt_under_original_title(self):
        text=CAPTION.replace('## 自然指令','## natural_instruction').replace('挥手','睁开眼晴')
        client=FakeSyncClient(semantic_reply(
            issues=[semantic_issue('自然指令','眼晴','改为眼睛','无歧义错别字')],
            corrected_sections={'自然指令':'0-5秒，<主体1>睁开眼睛。'}))
        report=qa.CaptionQAAgent(client=client,max_retries=0).check(text)
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertIn('睁开眼晴',payload['caption_sections']['自然指令'])
        self.assertNotIn('natural_instruction',payload['caption_sections'])
        self.assertEqual(report['revised_caption'],text.replace('眼晴','眼睛'))

    def test_async_semantic_check_gets_the_english_instruction_body(self):
        from captionqa.caption_qa_review import SafeAutoFixAgent
        text=CAPTION.replace('## 自然指令','## natural_instruction').replace(
            '<主体1>挥手。','<主体1>、<主体2>和<主体3>挥手。')
        client=FakeAsyncClient(semantic_reply())
        report=asyncio.run(SafeAutoFixAgent(async_client=client).acheck_single_caption(text))
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertIn('<主体3>挥手。',payload['caption_sections']['自然指令'])
        self.assertEqual(report['subject_check']['missing_subjects'],[])
        self.assertEqual(report['revised_caption'],text)

    def test_parser_history_is_invalidated_only_for_previously_unknown_aliases(self):
        english=CAPTION.replace('## 自然指令','## natural_instruction')
        chinese=CAPTION.replace('## 自然指令','## 自然描述')
        self.assertTrue(qa.instruction_parser_outdated(english,2))
        self.assertFalse(qa.instruction_parser_outdated(english,3))
        self.assertTrue(qa.instruction_parser_outdated(chinese,1))
        self.assertFalse(qa.instruction_parser_outdated(chinese,2))
        self.assertFalse(qa.instruction_parser_outdated(CAPTION,1))
        self.assertFalse(qa.instruction_parser_outdated(CAPTION+'备注中写 natural_instruction。',1))

    def test_natural_description_alias_with_complete_subjects_does_not_report_missing(self):
        complete=CAPTION.replace('<主体1>挥手。','<主体1>、<主体2>和<主体3>挥手。')
        for heading in ('## 自然描述\n','自然描述\n','##自然描述\n','## 自然描述：'):
            text=complete.replace('## 自然指令\n',heading)
            report=qa.CaptionQAAgent(dry_run=True).check(text)
            self.assertEqual(report['subject_check']['missing_in_instruction'],[])
            self.assertEqual(report['revised_caption'],text)
            self.assertEqual(report['caption_parser_version'],qa.CAPTION_PARSER_VERSION)

    def test_alias_still_reports_real_missing_subjects_and_yields_to_current_name(self):
        text=CAPTION.replace('## 自然指令','## 自然描述')
        report=qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report['subject_check']['missing_in_instruction'],['<主体2>','<主体3>'])
        selected=qa.parse_caption_sections(text+'## 自然指令\n<主体2>挥手。')
        self.assertEqual(selected['自然指令'],'<主体2>挥手。')
        self.assertTrue(selected.field_resolution_notes)

    def test_semantic_request_gets_alias_body_in_the_correct_field(self):
        text=CAPTION.replace('## 自然指令','## 自然描述')
        client=FakeSyncClient(semantic_reply())
        qa.CaptionQAAgent(client=client).check(text)
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertIn('<主体1>挥手。',payload['caption_sections']['自然指令'])
        self.assertNotIn('自然描述',payload['caption_sections'])

    def test_summary_cannot_cover_missing_instruction_subjects(self):
        report = qa.CaptionQAAgent(dry_run=True).check(CAPTION)
        coverage = report["subject_check"]
        self.assertEqual(coverage["summary_referenced_subjects"], ["<主体1>", "<主体2>", "<主体3>"])
        self.assertEqual(coverage["instruction_referenced_subjects"], ["<主体1>"])
        self.assertEqual(coverage["missing_in_instruction"], ["<主体2>", "<主体3>"])
        self.assertEqual(coverage["missing_subjects"], coverage["missing_in_instruction"])
        errors = report["static_errors"]
        self.assertEqual([i["original_snippet"] for i in errors], ["<主体2>", "<主体3>"])
        self.assertTrue(all(i["section"] == "自然指令" and not i["can_auto_fix"] and not i["resolved"] for i in errors))
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], CAPTION)
        self.assertIn("【重点：自然指令漏写已定义主体】", report["human_review_reason"])

    def test_all_instruction_subjects_present_passes_without_summary(self):
        text = CAPTION.replace("<主体1>挥手。", "<主体1>、<主体2>和<主体3>挥手。")
        text = text.replace("<主体1>、<主体2>和<主体3>站在一起。", "三人站在一起。")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["subject_check"]["missing_in_instruction"], [])

    def test_missing_or_empty_instruction_lists_every_defined_subject(self):
        for suffix in ("", "## 自然指令\n", "## 自然指令\n  \n"):
            with self.subTest(suffix=suffix):
                report = qa.CaptionQAAgent(dry_run=True).check(CAPTION.split("## 自然指令")[0] + suffix)
                self.assertEqual(report["subject_check"]["missing_in_instruction"], ["<主体1>", "<主体2>", "<主体3>"])

    def test_pronouns_do_not_count_as_explicit_subject_tags(self):
        text = CAPTION.replace("<主体1>挥手。", "他和她一起挥手，三人微笑。")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(len(report["subject_check"]["missing_in_instruction"]), 3)

    def test_bare_existing_tags_are_normalized_before_coverage(self):
        text = CAPTION.replace("<主体1>挥手。", "主体1、主体2和主体3挥手。")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["subject_check"]["missing_in_instruction"], [])
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("<主体1>、<主体2>和<主体3>挥手。", report["revised_caption"])

    def test_subject_numbers_match_exactly_and_ghost_checks_are_preserved(self):
        text = CAPTION.replace("<主体1>挥手。", "<主体10>挥手。")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["subject_check"]["missing_in_instruction"], ["<主体1>", "<主体2>", "<主体3>"])
        self.assertEqual(report["subject_check"]["ghost_subjects"], ["<主体10>"])

    def test_plain_alias_and_inline_instruction_headers_share_the_same_rule(self):
        for heading in ("自然指令（细节）\n", "##自然指令(细节)\n", "## 自然指令    "):
            text = CAPTION.replace("## 自然指令\n", heading)
            report = qa.CaptionQAAgent(dry_run=True).check(text)
            self.assertEqual(report["subject_check"]["missing_in_instruction"], ["<主体2>", "<主体3>"])
            self.assertEqual(report["revised_caption"], text)

    def test_scene_and_picture_elements_do_not_join_subject_coverage(self):
        sections = {"主体": "<主体1>：白衣女子。", "自然指令": "<主体1>挥手。",
                    "场景": "<场景1>：广场。", "画面元素": "<画面元素2>：长椅。"}
        self.assertEqual(qa.run_static_rules(sections), [])

    def test_model_gets_explicit_policy_and_cannot_silence_static_alerts(self):
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(CAPTION)
        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertTrue(payload["auto_fix_policy"]["require_defined_subjects_in_natural_instruction"])
        self.assertEqual(payload["subject_coverage"]["missing_in_instruction"], ["<主体2>", "<主体3>"])
        self.assertEqual(len(payload["static_errors"]), 2)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")

    def test_model_cannot_auto_insert_the_missing_tags_or_actions(self):
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("自然指令", "<主体1>挥手。", "补全主体", "标点符号硬伤")],
            corrected_sections={"自然指令": "0-5秒，<主体1>挥手，<主体2>和<主体3>走来。"}))
        report = qa.CaptionQAAgent(client=client).check(CAPTION)
        self.assertEqual(report["revised_caption"], CAPTION)
        self.assertEqual(report["subject_check"]["missing_in_instruction"], ["<主体2>", "<主体3>"])
        self.assertTrue(any(i["issue_type"] == "unsafe_llm_edit" for i in report["issues"]))

    def test_problem_id_once_and_prominent_explanation_with_exact_missing_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.jsonl", root / "ids.txt"
            good = CAPTION.replace("<主体1>挥手。", "<主体1>、<主体2>和<主体3>挥手。")
            write_jsonl(source, [{"id": "0001", "caption": CAPTION}, {"id": "0002", "caption": good, "备注": CAPTION}])
            original = source.read_bytes()
            asyncio.run(qa.aprocess_batch(source, output, CountingAgent()))
            self.assertEqual(output.read_text(), "0001\n")
            self.assertEqual(source.read_bytes(), original)
            explanation = output.with_name("ids_检测说明.txt").read_text()
            self.assertIn("【重点复核：自然指令漏写已定义主体】", explanation)
            self.assertIn("ID 0001（原表第 2 行）：自然指令缺少 <主体2>、<主体3>。", explanation)
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["instruction_subject_alerts"], [
                {"id": "0001", "excel_row": 2, "missing_subjects": ["<主体2>", "<主体3>"]}])
            self.assertEqual(read_jsonl(output.with_suffix(".jsonl"))[0]["qa_status"], "NEED_HUMAN_REVIEW")

    def test_markdown_report_emphasizes_instruction_specific_coverage(self):
        stream = io.StringIO()
        qa.print_markdown_report(qa.CaptionQAAgent(dry_run=True).check(CAPTION), stream=stream)
        self.assertIn("自然指令漏写已定义主体（重点复核）：<主体2>、<主体3>", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
