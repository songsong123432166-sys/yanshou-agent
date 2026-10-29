"""The supplied symbol specification must not rewrite speech or screen text."""
import json
from pathlib import Path
import tempfile
import unittest

import caption_qa_agent as qa
from caption_qa_repair_table import read_repair_entries
from test_caption_qa_upgrades import GOOD, FakeSyncClient, semantic_issue, semantic_reply


class SymbolPolicyTests(unittest.TestCase):
    def caption(self, text):
        return GOOD.replace('0-4秒，第0秒，<主体1>向前跑。', '<主体1>站立。' + text)

    def test_user_symbol_examples_and_music_with_nested_vocals_are_valid(self):
        examples = (
            '（背景音乐是韩语流行歌曲《You(=I)》，C调，四四拍，速度在108BPM左右，钢琴以柱式和弦为主，节奏跳跃，一个幼态发嗲的青年女声演唱道{오늘은 안된다고 말하지마 오늘만큼은 내게도 꼭}，营造出可爱俏皮的氛围）。',
            '画外<传来噼啪的燃烧声>。', '<环境中有持续的吱吱蝉鸣声>。',
            '（背景中继续播放着<视频1>中的纯音乐，钢琴弹奏舒缓悠扬旋律，整体旋律更饱满，营造出平静且略带伤感的氛围）。',
            '刻有白色大写英文【LAUNDRY】的透明亚克力标牌。',
        )
        for text in examples:
            with self.subTest(text=text):
                raw = self.caption(text)
                report = qa.CaptionQAAgent(dry_run=True).check(raw)
                self.assertEqual(report['status'], 'PASS')
                self.assertEqual(report['revised_caption'], raw)

    def test_mismatched_prescribed_symbols_are_manual_and_never_guessed(self):
        for text in ('说道{你好。', '出现【LAUNDRY。', '（背景播放纯音乐。', '<传来燃烧声。', '出现【文字}。'):
            with self.subTest(text=text):
                raw = self.caption(text)
                report = qa.CaptionQAAgent(dry_run=True).check(raw)
                self.assertEqual(report['status'], 'NEED_HUMAN_REVIEW')
                self.assertTrue(any(i['issue_type'] == 'symbol_format' for i in report['issues']))
                self.assertEqual(report['revised_caption'], raw)

    def test_literal_spoken_and_displayed_words_are_not_entity_tags(self):
        raw = self.caption('说道{我是主体1，3<5}，牌子上写着【主体2 < 主体3 >】。')
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['revised_caption'], raw)
        self.assertEqual(report['subject_check']['ghost_subjects'], [])

    def test_quoted_subject_label_cannot_replace_actual_instruction_reference(self):
        raw = GOOD.replace('0-4秒，第0秒，<主体1>向前跑。', '男子说道{<主体1>}，举着【<主体1>】标牌。')
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report['subject_check']['missing_in_instruction'], ['<主体1>'])

    def test_multiline_foreign_speech_and_visible_text_are_preserved(self):
        raw = self.caption('演唱道{오늘은\n안된다고}，标牌写着【LAUNDRY\nOPEN】。')
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['revised_caption'], raw)

    def test_unclosed_literal_is_flagged_without_editing_its_words_or_ending(self):
        for text in ('说道{我是主体2', '标牌写着【主体3'):
            with self.subTest(text=text):
                raw = self.caption(text)
                report = qa.CaptionQAAgent(dry_run=True).check(raw)
                self.assertEqual(report['status'], 'NEED_HUMAN_REVIEW')
                self.assertEqual(report['revised_caption'], raw)
                self.assertEqual(report['subject_check']['ghost_subjects'], [])

    def test_symbol_category_change_is_review_only_even_if_model_requests_auto_fix(self):
        old = '<主体1>说道【你好】。'
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue('自然指令', old, '人声应使用大括号', '内容符号不规范')],
            corrected_sections={'自然指令': '<主体1>说道{你好}。'}))
        raw = GOOD.replace('0-4秒，第0秒，<主体1>向前跑。', old)
        report = qa.CaptionQAAgent(client=client, max_retries=0).check(raw)
        self.assertEqual(report['revised_caption'], raw)
        self.assertFalse(report['llm_issues'][0]['can_auto_fix'])
        payload = json.loads(client.calls[0]['messages'][1]['content'])
        self.assertEqual(payload['auto_fix_policy']['content_symbols']['文字'], '【】')

    def test_model_cannot_disguise_content_bracket_changes_as_punctuation(self):
        old = '<主体1>说道【你好】。'
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue('自然指令', old, '更换括号', '标点符号硬伤')],
            corrected_sections={'自然指令': '<主体1>说道{你好}。'}))
        raw = GOOD.replace('0-4秒，第0秒，<主体1>向前跑。', old)
        report = qa.CaptionQAAgent(client=client, max_retries=0).check(raw)
        self.assertEqual(report['revised_caption'], raw)
        self.assertTrue(any(i['issue_type'] == 'unsafe_llm_edit' for i in report['issues']))

    def test_report_combines_subject_tags_without_treating_them_as_task_labels(self):
        raw = GOOD.replace('## 总结', '<主体2>：女子。\n<主体3>：儿童。\n## 总结')
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        row = {'row_index': 1, 'id': '0001', 'caption': raw, 'qa_scope': 'static_only',
               'qa_report': report, 'qa_checkpoint': {'completed': True, 'caption_field': 'caption'}}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache.jsonl'
            cache.write_text(json.dumps(row) + '\n')
            entries, missing = read_repair_entries(cache, id_column='id', dry_run=True)
        self.assertEqual(missing, [])
        self.assertEqual(len(entries), 1)
        self.assertIn('<主体2>、<主体3>', entries[0][1])
        self.assertEqual(entries[0][1].count('自然指令漏写已定义主体'), 1)

    def test_task_and_audiovisual_labels_accept_both_vertical_bars_and_keep_internal_slashes(self):
        for field in qa.MULTI_LABEL_SECTIONS:
            for value in ('人物参考|动态参考', '实体/IP｜动态', '实体/IP - 相同｜镜头/运镜 - 同类不同个', '人物参考'):
                with self.subTest(field=field, value=value):
                    sections = {field: value}
                    self.assertEqual(qa.run_static_rules(sections), [])
                    self.assertEqual(qa.apply_static_fixes(sections), sections)

    def test_task_label_delimiter_errors_are_manual_not_metadata_rewrites(self):
        for value in ('人物参考、动态参考', '人物参考，动态参考', '人物参考\n动态参考', '人物参考||动态参考'):
            with self.subTest(value=value):
                sections = {'任务类型': value}
                self.assertEqual([i['issue_type'] for i in qa.run_static_rules(sections)], ['label_separator'])
                self.assertEqual(qa.apply_static_fixes(sections), sections)

    def test_audio_reference_tags_are_supported_without_changing_spoken_words(self):
        raw = self.caption('与音频1相同的背景音乐，< 音频2 >中同款音效，演唱道{音频3}。')
        report = qa.CaptionQAAgent(dry_run=True).check(raw)
        self.assertIn('与<音频1>相同', report['revised_caption'])
        self.assertIn('<音频2>中同款', report['revised_caption'])
        self.assertIn('演唱道{音频3}', report['revised_caption'])

    def test_unclear_legacy_heading_boundary_is_not_misreported_as_a_label_separator(self):
        sections = {'similarity_tag': '实体/IP-相同|动态\n\n###audio_similarity_tag\n音色-相同'}
        self.assertEqual(qa.run_static_rules(sections), [])
        self.assertEqual(qa.apply_static_fixes(sections), sections)


if __name__ == '__main__':
    unittest.main()
