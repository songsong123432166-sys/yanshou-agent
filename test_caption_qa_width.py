"""Width rules change prescribed symbols without changing protected content."""
import asyncio
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_review import SafeAutoFixAgent, AUTO_WORKFLOW, PREVIOUS_AUTO_WORKFLOW, make_review, approved_changes, summary
from caption_qa_review_io import export_approved
from caption_qa_review_report import highlight_ranges
from caption_qa_width import normalize, operations
from test_caption_qa_upgrades import GOOD, FakeSyncClient, semantic_reply


class WidthTests(unittest.TestCase):
    def caption(self,text):
        return GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text)

    def check(self,text):
        return SafeAutoFixAgent(dry_run=True).check(self.caption(text))

    def test_chinese_prose_marks_and_terminal_dot_are_normalized_without_extra_period(self):
        report=self.check('<主体1>转头, 看向镜头; 然后询问: 你好吗? 太好了! 最后转身.')
        self.assertIn('<主体1>转头， 看向镜头； 然后询问： 你好吗？ 太好了！ 最后转身。',report['revised_caption'])
        self.assertNotIn('。。',report['revised_caption'])
        self.assertEqual(report['status'],'AUTO_FIXED')
        self.assertTrue(all(i['resolved'] for i in report['issues']))

    def test_full_width_tag_definitions_and_references_do_not_create_subject_false_positives(self):
        raw=GOOD.replace('<主体1>','＜主体1＞')
        report=SafeAutoFixAgent(dry_run=True).check(raw)
        self.assertEqual(report['revised_caption'],GOOD)
        self.assertEqual(report['subject_check']['missing_in_instruction'],[])
        self.assertEqual(report['subject_check']['ghost_subjects'],[])
        self.assertEqual(report['status'],'AUTO_FIXED')

    def test_tag_width_and_internal_spaces_are_independent_nonconflicting_edits(self):
        report=self.check('＜ 主体1 ＞向前跑.')
        self.assertIn('<主体1>向前跑。',report['revised_caption'])
        self.assertEqual(report['status'],'AUTO_FIXED')

    def test_music_outer_parentheses_only_preserve_all_inner_symbols_and_song_titles(self):
        for audio in ['(背景音乐是《You(=I)》, 钢琴轻快)', '(背景音乐淡出）', '（背景音乐渐强)']:
            report=self.check('<主体1>站立。'+audio)
            expected='（'+audio[1:-1]+'）。'
            self.assertIn(expected,report['revised_caption'])
            self.assertEqual(report['status'],'AUTO_FIXED')

    def test_voice_and_sound_wrapper_width_changes_leave_audio_interior_untouched(self):
        for old,new in [('｛Hello, world! 主体2｝','{Hello, world! 主体2}'),
                        ('＜传来噼啪, 的燃烧声＞','<传来噼啪, 的燃烧声>')]:
            report=self.check('<主体1>说道'+old)
            self.assertIn(new+'。',report['revised_caption'])
            self.assertEqual(report['status'],'AUTO_FIXED')
        nested='(背景音乐是《You(=I)》, 演唱道｛Hi, hi!｝)'
        self.assertIn('（'+nested[1:-1]+'）。',self.check('<主体1>站立。'+nested)['revised_caption'])

    def test_screen_words_foreign_quotes_and_ordinary_parentheses_are_preserved(self):
        text='<主体1>举牌【Hello, world! ＜主体2＞】, 说完“Hello, world!”后转头(备注).'
        report=self.check(text)
        self.assertIn('【Hello, world! ＜主体2＞】',report['revised_caption'])
        self.assertIn('“Hello, world!”',report['revised_caption'])
        self.assertIn('(备注)。',report['revised_caption'])
        self.assertEqual(normalize('自然指令','普通比较a＜b，和函数f(x)，曲名You(=I)。'),'普通比较a＜b，和函数f(x)，曲名You(=I)。')

    def test_numbers_times_urls_files_and_english_prose_are_preserved(self):
        text='<主体1>查看https://example.com/a?q=1.5，时间01:02, 镜头在1.5秒时移动。文件a.csv、目录/tmp/a.csv保持。Hello, world!'
        report=self.check(text)
        fixed=report['revised_caption']
        for token in ['https://example.com/a?q=1.5','01:02','1.5秒','a.csv','/tmp/a.csv','Hello, world!']:
            self.assertIn(token,fixed)
        self.assertIn('时间01:02， 镜头',fixed)

    def test_metadata_both_pipes_slashes_and_full_width_letters_are_not_nfkc_converted(self):
        for name in qa.PUNCTUATION_EXEMPT_SECTIONS:
            value='实体/IP｜动态|人物参考, ＡＢＣ'
            self.assertEqual(normalize(name,value,exempt=True),value)
        self.assertEqual(normalize('自然指令','<主体1>拿着ＡＢＣ１２３标牌。'),'<主体1>拿着ＡＢＣ１２３标牌。')

    def test_incomplete_wrapper_remains_manual_without_inventing_or_editing_inner_words(self):
        for text in ['<主体1>说道｛Hello, world!', '<主体1>站立。(背景音乐是钢琴, 轻快', '＜主体1向前跑。']:
            report=self.check(text)
            self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')
            self.assertIn(text,report['revised_caption'])
            self.assertTrue(any(i['issue_type']=='symbol_width' and not i['resolved'] for i in report['issues']))

    def test_normalization_is_idempotent_and_model_receives_normalized_caption(self):
        raw=self.caption('＜主体1＞转头, 看向右侧.')
        client=FakeSyncClient(semantic_reply())
        report=SafeAutoFixAgent(client=client).check(raw)
        sent=json.loads(client.calls[0]['messages'][1]['content'])['caption_sections']
        self.assertIn('<主体1>转头， 看向右侧。',sent['自然指令'])
        again=SafeAutoFixAgent(dry_run=True).check(report['revised_caption'])
        self.assertEqual(again['revised_caption'],report['revised_caption'])
        self.assertEqual(again['status'],'PASS')
        async_report=asyncio.run(SafeAutoFixAgent(dry_run=True).acheck_single_caption(raw))
        self.assertEqual(async_report['revised_caption'],report['revised_caption'])

    def test_previous_workflow_keeps_its_policy_for_checkpoint_continuation(self):
        agent=SafeAutoFixAgent(dry_run=True,normalize_width=False)
        self.assertEqual(agent.workflow_version,PREVIOUS_AUTO_WORKFLOW)
        report=agent.check(self.caption('<主体1>转头, 看向右侧。'))
        self.assertIn('转头, 看向右侧。',report['revised_caption'])
        self.assertEqual(report['workflow_version'],PREVIOUS_AUTO_WORKFLOW)

    def test_width_wrappers_export_with_audit_but_forged_audio_inner_operation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.jsonl';cache=root/'cache.jsonl'
            raw=self.caption('<主体1>站立。(背景音乐是钢琴, 轻快)')
            source.write_text(json.dumps({'id':'0007','caption':raw,'note':'不改其他列'})+'\n')
            asyncio.run(qa.aprocess_batch(source,cache,SafeAutoFixAgent(dry_run=True),column='caption'))
            doc=make_review(cache,source,source,caption_column='caption',id_column='id')
            before=source.read_bytes();result=export_approved(doc,source,source,cache,root,'data')
            value=json.loads(Path(result['corrected_path']).read_text())
            self.assertIn('（背景音乐是钢琴, 轻快）。',value['caption'])
            self.assertEqual(value['id'],'0007');self.assertEqual(value['note'],'不改其他列')
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(summary(doc)['pending'],0)
            corrupt=copy.deepcopy(doc)
            issue=next(i for i in corrupt['issues'] if any(o.get('width_rule') for o in i['operations']))
            field=next(f for f in corrupt['fields'] if f['field_id']==issue['field_id'])
            pos=field['original'].index(',')
            issue['operations']=[{'start':pos,'end':pos+1,'old':',','new':'，','width_rule':'背景音乐外层括号'}]
            with self.assertRaisesRegex(ValueError,'音频保护'):
                approved_changes(corrupt)

    def test_default_web_generates_width_copy_and_keeps_source_and_real_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);app=web.App(root/'out',root/'jobs');self.addCleanup(app.close)
            raw=self.caption('＜主体1＞转头, 看向右侧.')
            data=(json.dumps({'id':'0007','caption':raw,'note':'原样保留'})+'\n').encode()
            with patch('caption_qa_review_report.export_review',return_value={}),patch.object(web,'read_saved_key',side_effect=AssertionError('offline')):
                item=app.create_job('width.jsonl',io.BytesIO(data),len(data),dry_run=True)
                app.executor.submit(lambda:None).result(timeout=10)
            job=app.jobs[item['id']]
            self.assertEqual(job.workflow_version,AUTO_WORKFLOW)
            self.assertEqual(job.status,'done',job.error)
            value=json.loads(Path(job.snapshot()['review_output']['corrected_path']).read_text())
            self.assertIn('<主体1>转头， 看向右侧。',value['caption'])
            self.assertEqual(value['id'],'0007');self.assertEqual(value['note'],'原样保留')
            self.assertEqual(job.source.read_bytes(),data)
            document=app.review(job.id)['review']
            for issue in document['issues']:
                if issue['kind']=='symbol_width':
                    excerpt='\n'.join(issue['excerpts'])
                    self.assertEqual([excerpt[a:b] for a,b in highlight_ranges(excerpt,issue)],[issue['snippet']])


if __name__=='__main__':unittest.main()
