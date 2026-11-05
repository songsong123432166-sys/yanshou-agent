"""Whitespace-only separator changes, with opaque content left byte-for-byte."""
import unittest
from captionqa.caption_qa_spacing import normalize
from captionqa.caption_qa_labels import normalize_separators

class SpacingTests(unittest.TestCase):
    def test_time_hyphens_have_exactly_one_ascii_space_and_are_idempotent(self):
        for source in ['0-2秒','0 -2秒','0-  2秒','0\t-\t2秒','0　-　2秒','0 - 2秒']:
            text=source+'，镜头前推。'
            self.assertEqual(normalize('自然指令',text),text)
            self.assertEqual(normalize('自然指令',normalize('自然指令',text)),normalize('自然指令',text))
        self.assertEqual(normalize('自然指令','0.5-2.5秒，转头。00:02-00:08，奔跑。'),'0.5-2.5秒，转头。00:02-00:08，奔跑。')

    def test_metadata_separator_spacing_does_not_change_slashes_or_pipe_width(self):
        text='实体/IP-相同|镜头/运镜 -相同｜编辑-改'
        self.assertEqual(normalize('video_similarity_tag',text),'实体/IP-相同|镜头/运镜-相同｜编辑-改')
        self.assertEqual(normalize('任务类型','参考-严格编辑-改'),'参考-严格编辑-改')
        self.assertEqual(normalize('video_similarity_tag','实体-T-shirt｜物品-GPT-4'),'实体-T-shirt｜物品-GPT-4')
        self.assertEqual(normalize_separators('任务类型','参考 - 严格编辑 - 改，人物参考'),'参考-严格编辑-改|人物参考')

    def test_audio_screen_quotes_english_negative_numbers_and_urls_are_not_changed(self):
        for text in ['<主体1>说道{0-2秒，go-go}。','（背景音乐0-2秒，继续）。','<0-2秒，传来脚步声>。',
                     '显示【0-2秒】。','写着“0-2秒”。','T-shirt，AI-generated，-3，-2-1秒。',
                     '地址https://example.test/0-2s，原样。','路径 /tmp/0-2s，原样。']:
            self.assertEqual(normalize('自然指令',text),text)

    def test_time_formatting_does_not_touch_point_events_or_other_range_symbols(self):
        text='0-2秒，第1秒，转头。2～8秒，第4秒，微笑。'
        self.assertEqual(normalize('自然指令',text),text)
        self.assertEqual(normalize('similarity_label','Yes-No'),'Yes-No')

if __name__=='__main__':unittest.main()
