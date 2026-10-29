"""Current field precedence and read-only model context preserve source data."""
import asyncio
import csv
import io
import json
import unittest
from unittest.mock import patch

import captionqa.caption_qa_agent as qa
import captionqa.caption_qa_web as web
from captionqa.caption_field_schema import normalize, render, FIELDS
from tests.test_caption_qa_upgrades import GOOD, FakeSyncClient, semantic_reply
import tests.test_caption_qa_web as web_tests


class FieldPriorityTests(unittest.TestCase):
    def test_current_subject_wins_in_both_physical_orders_without_old_ghosts(self):
        old='## subject\n<主体8>：旧版本人物。\n'
        for text in [old+GOOD,GOOD+old]:
            with self.subTest(text=text):
                report=qa.CaptionQAAgent(dry_run=True).check(text)
                self.assertEqual(report['subject_check']['defined_subjects'],['<主体1>'])
                self.assertEqual(report['subject_check']['ghost_subjects'],[])
                self.assertEqual(report['revised_caption'],text)
                self.assertTrue(report['field_resolution_notes'])

    def test_empty_current_subject_falls_back_to_old_but_explicit_none_is_authoritative(self):
        legacy='## subject\n<主体1>：男子。\n'
        empty=GOOD.replace('<主体1>：身穿白袍。','')+legacy
        self.assertEqual(qa.parse_caption_sections(empty)['主体'].strip(),'<主体1>：男子。')
        explicit=GOOD.replace('<主体1>：身穿白袍。','无。')+legacy
        report=qa.CaptionQAAgent(dry_run=True).check(explicit)
        self.assertEqual(report['subject_check']['defined_subjects'],[])
        self.assertEqual(report['subject_check']['ghost_subjects'],['<主体1>'])

    def test_two_current_subject_fields_remain_ambiguous(self):
        with self.assertRaises(ValueError):qa.parse_caption_sections(GOOD+'## 主体\n<主体2>：另一人。\n')

    def test_model_receives_frame_definition_as_read_only_context(self):
        text=GOOD.replace('<主体1>向前跑。','<主体1>向前跑，保留<画面元素1>。')+'## 画面元素\n<画面元素1>：右上角【bilibili】水印。\n'
        client=FakeSyncClient(semantic_reply())
        report=qa.CaptionQAAgent(client=client).check(text)
        data=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertIn('<画面元素1>',data['caption_sections']['画面元素'])
        self.assertEqual(data['read_only_sections'],['画面元素'])
        self.assertNotIn('画面元素',data['checked_sections'])
        self.assertEqual(report['revised_caption'],text)
        attack=FakeSyncClient(semantic_reply(corrected_sections={'画面元素':'无。'}))
        self.assertEqual(qa.CaptionQAAgent(client=attack,max_retries=0).check(text)['revised_caption'],text)

    def test_priority_and_context_invalidate_only_affected_old_results(self):
        self.assertTrue(qa.instruction_parser_outdated(GOOD+'## subject\n旧版本。\n',4))
        frame=GOOD+'## 画面元素\n<画面元素1>：水印。\n'
        self.assertTrue(qa.instruction_parser_outdated(frame,4,semantic=True))
        self.assertFalse(qa.instruction_parser_outdated(frame,4,semantic=False))
        self.assertFalse(qa.instruction_parser_outdated(GOOD,4,semantic=True))

    def test_normalizer_uses_current_values_and_keeps_original_in_audit(self):
        current=render({field:('人物参考' if field=='任务类型' else '<主体1>：男子。' if field=='主体' else '') for field in FIELDS})
        raw=current+'## subject\n<主体8>：旧版本。\n'
        result=asyncio.run(normalize(raw))
        self.assertEqual(result['caption'],current)
        self.assertEqual(result['original'],raw)
        self.assertEqual(result['model_requests'],0)
        self.assertIn('新版字段优先',result['tags'])

    def test_normalizer_can_fall_back_to_old_prompt_with_current_task(self):
        current=render({field:('人物参考' if field=='任务类型' else '') for field in FIELDS})
        for text in [current,current.replace('## 提示\n\n','')]:
            result=asyncio.run(normalize(text+'## subtask_type\n以<图片1>为基础，生成视频。\n'))
            self.assertEqual(result['status'],'converted')
            self.assertIn('## 提示\n以<图片1>为基础，生成视频。',result['caption'])
            self.assertEqual(result['model_requests'],0)


class FieldPriorityWebTests(unittest.TestCase):
    setUp=web_tests.WebTests.setUp
    tearDown=web_tests.WebTests.tearDown
    finish=web_tests.WebTests.finish

    def test_active_field_edits_preserve_inactive_old_field_and_other_columns(self):
        text=GOOD.replace('身穿白袍。','身穿白袍')+'## subject\n<主体8>：旧版本。\n'
        data=web_tests.csv_bytes([('0001',text)])
        with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
            item=self.app.create_job('priority.csv',io.BytesIO(data),len(data),dry_run=True)
            self.finish()
        job=self.app.jobs[item['id']];self.assertEqual(job.status,'done',job.error)
        self.assertEqual(job.source.read_bytes(),data)
        self.assertTrue(job.snapshot()['field_resolution_notes'])
        result=job.id_summary['review_output']
        with open(result['corrected_path'],encoding='utf-8-sig',newline='') as stream:rows=list(csv.reader(stream))
        self.assertIn('## subject\n<主体8>：旧版本。\n',rows[1][1])
        self.assertIn('<主体1>：身穿白袍。',rows[1][1])
        self.assertEqual(rows[1][0],'0001')


if __name__=='__main__':unittest.main()
