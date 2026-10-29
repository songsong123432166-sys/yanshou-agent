"""Legacy field ownership and fresh recheck jobs; no real keys or API calls."""
import asyncio
import io
import json
import unittest
from unittest.mock import patch

import captionqa.caption_qa_agent as qa
import captionqa.caption_qa_web as web
from captionqa.caption_field_schema import is_generation_request
from captionqa.caption_qa_review import SafeAutoFixAgent
from tests.test_caption_qa_upgrades import FakeAsyncClient, semantic_reply
import tests.test_caption_qa_web as web_tests


PROMPT='以<图片1>中的机甲界面为基础，保留特写，生成一段科幻视频。'
OLD=('## task_type\n人物参考｜视觉风格参考\n\n## subtask_type\n'+PROMPT+
     '\n\n## brief_captions\n<图片1>：机甲界面。\n\n## similarity_tag\n实体/IP-相同\n\n'
     '## subject\n<主体1>：机器人。\n<主体2>：飞船。\n\n'
     '## 总结\n<主体1>追逐<主体2>。\n\n## natural_instruction\n'
     '0-4秒，<主体1>飞向<主体2>。\n')


class LegacyFieldTests(unittest.TestCase):
    def test_prompt_prose_and_subject_alias_are_checked_as_their_actual_fields(self):
        result=qa.CaptionQAAgent(dry_run=True).check(OLD)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['subject_check']['defined_subjects'],['<主体1>','<主体2>'])
        self.assertEqual(result['subject_check']['ghost_subjects'],[])
        self.assertEqual(result['revised_caption'],OLD)
        sections=qa.parse_caption_sections(OLD)
        self.assertEqual(sections['提示'].strip(),PROMPT)
        self.assertIn('输入描述',sections);self.assertIn('video_similarity_tag',sections)

    def test_prompt_and_subject_are_in_async_model_request_and_punctuation_is_preserved(self):
        client=FakeAsyncClient(semantic_reply())
        result=asyncio.run(SafeAutoFixAgent(async_client=client).acheck_single_caption(OLD))
        payload=json.loads(client.calls[0]['messages'][1]['content'])['caption_sections']
        self.assertEqual(payload['提示'],PROMPT);self.assertIn('<主体1>',payload['主体'])
        self.assertEqual(result['revised_caption'],OLD)

    def test_real_subtask_labels_still_get_separator_checks(self):
        for value in ['人物参考，视觉风格参考','编辑-改','延长-向后延长','剪辑/卡点迁移']:
            with self.subTest(value=value):
                self.assertFalse(is_generation_request(value))
                text=OLD.replace(PROMPT,value);sections=qa.parse_caption_sections(text)
                self.assertIn('subtask_type',sections);self.assertNotIn('提示',sections)
                errors=qa.run_static_rules(sections)
                self.assertEqual(any(i['issue_type']=='label_separator' for i in errors),'，' in value)

    def test_existing_current_prompt_and_subject_take_precedence_and_original_is_preserved(self):
        for suffix in ['## 提示\n生成另一段视频。\n','## 主体\n<主体9>：另一个人。\n']:
            result=qa.CaptionQAAgent(dry_run=True).check(OLD+suffix)
            self.assertEqual(result['qa_scope'],'static_only')
            self.assertTrue(result['field_resolution_notes'])
            self.assertEqual(result['revised_caption'],OLD+suffix)

    def test_legacy_header_versions_are_targeted_not_global(self):
        self.assertTrue(qa.instruction_parser_outdated(OLD,3))
        self.assertFalse(qa.instruction_parser_outdated(OLD,5))
        canonical=OLD.replace('## subtask_type','## 提示').replace('## brief_captions','## 输入描述').replace(
            '## subject','## 主体').replace('## similarity_tag','## video_similarity_tag').replace(
            '## natural_instruction','## 自然指令').replace('## task_type','## 任务类型')
        self.assertFalse(qa.instruction_parser_outdated(canonical,1))


class RecheckTests(unittest.TestCase):
    setUp=web_tests.WebTests.setUp
    tearDown=web_tests.WebTests.tearDown
    finish=web_tests.WebTests.finish
    request=web_tests.WebTests.request
    def test_recheck_creates_fresh_job_and_keeps_reviews_and_source(self):
        data=web_tests.csv_bytes([('0001',OLD.replace('机器人。','机器人'))])
        created=self.app.create_job('legacy.csv',io.BytesIO(data),len(data),dry_run=True)
        self.finish();old=self.app.jobs[created['id']]
        review=self.app.review(old.id);field=review['review']['fields'][0]
        self.app.review(old.id,{'revision':review['review']['revision'],'field_id':field['field_id'],
                                'final_text':field['original'].strip()})
        paths=[old.source,old.cache,old.output,old.cache.with_suffix('.review.json')]
        original={p:p.read_bytes() for p in paths}
        with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
            status,body=self.request('POST',f'/api/jobs/{old.id}/recheck',token=False)
            self.assertEqual(status,403)
            status,body=self.request('POST',f'/api/jobs/{old.id}/recheck')
            self.assertEqual(status,202,body);self.finish()
        fresh=self.app.jobs[json.loads(body)['id']]
        self.assertNotEqual(fresh.id,old.id);self.assertEqual(fresh.recheck_of,old.id)
        self.assertEqual(fresh.status,'done',fresh.error)
        self.assertEqual(fresh.source.read_bytes(),data)
        self.assertNotEqual(fresh.cache,old.cache);self.assertEqual(fresh.dry_run,old.dry_run)
        self.assertEqual(self.app.review(fresh.id)['summary']['manual_fields'],0)
        for path,content in original.items():self.assertEqual(path.read_bytes(),content)
        self.assertEqual(json.loads(fresh.cache.read_text())['qa_report']['subject_check']['ghost_subjects'],[])

    def test_semantic_recheck_calls_fake_model_again_instead_of_reusing_cache(self):
        data=web_tests.csv_bytes([('0001',OLD)])
        clients=[FakeAsyncClient(semantic_reply()),FakeAsyncClient(semantic_reply())]
        original_agent=qa.CaptionQAAgent
        def agent(**options):return original_agent(async_client=clients.pop(0),**options)
        first_client,second_client=clients
        with patch.object(web,'read_saved_key',return_value='offline-test'),patch.object(qa,'CaptionQAAgent',side_effect=agent):
            created=self.app.create_job('semantic.csv',io.BytesIO(data),len(data),review_first=False)
            self.finish();fresh=self.app.recheck(created['id']);self.finish()
        self.assertFalse(fresh['dry_run'])
        self.assertEqual(len(first_client.calls),1);self.assertEqual(len(second_client.calls),1)
        self.assertEqual(self.app.jobs[fresh['id']].status,'done')

    def test_active_recheck_is_reused_and_running_source_cannot_be_rechecked(self):
        data=web_tests.csv_bytes([('0001',OLD)])
        created=self.app.create_job('legacy.csv',io.BytesIO(data),len(data),dry_run=True)
        self.finish();old=self.app.jobs[created['id']]
        with patch.object(self.app.executor,'submit'):
            fresh=self.app.recheck(old.id);again=self.app.recheck(old.id)
            self.assertEqual(fresh['id'],again['id'])
            with self.assertRaises(ValueError):self.app.recheck(fresh['id'])


if __name__=='__main__':unittest.main()
