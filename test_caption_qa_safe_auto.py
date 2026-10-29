"""Default whitelist execution, protected content and human-review boundaries."""
import asyncio
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_review import SafeAutoFixAgent, AUTO_WORKFLOW, make_review, approved_changes, summary
from caption_qa_review_report import HEADERS, import_review
from test_caption_qa_upgrades import GOOD, FakeSyncClient, semantic_reply, semantic_issue


class SafeAutoTests(unittest.TestCase):
    def test_static_punctuation_and_existing_tag_are_automatic_sync_and_async(self):
        raw=GOOD.replace('<主体1>向前跑。','主体1向前跑')
        expected=GOOD
        for report in [SafeAutoFixAgent(dry_run=True).check(raw),
                       asyncio.run(SafeAutoFixAgent(dry_run=True).acheck_single_caption(raw))]:
            self.assertEqual(report['revised_caption'],expected)
            self.assertEqual(report['status'],'AUTO_FIXED')
            self.assertFalse(report['need_human_review'])
            self.assertTrue(all(i['resolved'] for i in report['issues']))
            self.assertEqual(report['automatic_issue_count'],2)

    def test_same_field_subject_issue_remains_pending_while_period_is_fixed(self):
        raw=GOOD.replace('<主体1>向前跑。','男子向前跑')
        report=SafeAutoFixAgent(dry_run=True).check(raw)
        self.assertIn('男子向前跑。',report['revised_caption'])
        self.assertNotIn('<主体1>向前跑',report['revised_caption'])
        self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')
        self.assertFalse(next(i for i in report['issues'] if i['issue_type']=='missing_subjects')['resolved'])
        self.assertTrue(next(i for i in report['issues'] if i['issue_type']=='missing_punctuation')['resolved'])

    def test_audio_interiors_unchanged_but_outer_period_is_automatic(self):
        for audio in ['<主体1>说道{你好,,朋友}',
                      '（背景音乐是钢琴,,轻快）', '<环境中持续的吱吱,,蝉鸣声>']:
            raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',audio)
            report=SafeAutoFixAgent(dry_run=True).check(raw)
            self.assertIn(audio+'。',report['revised_caption'])

    def test_known_typo_and_non_audio_punctuation_are_automatic(self):
        for old,new,kind in [('眼晴','眼睛','无歧义错别字'),
                             ('眉头紧锁、身穿白袍','眉头紧锁，身穿白袍','标点符号硬伤')]:
            raw=GOOD.replace('身穿白袍',old)
            proposed=qa.parse_caption_sections(raw)['主体'].replace(old,new)
            client=FakeSyncClient(semantic_reply(issues=[semantic_issue('主体',old,new,kind)],
                                                 corrected_sections={'主体':proposed.strip()}))
            report=SafeAutoFixAgent(client=client).check(raw)
            self.assertEqual(report['revised_caption'],raw.replace(old,new))
            self.assertEqual(report['status'],'AUTO_FIXED')

    def test_model_audio_punctuation_not_executed_but_outer_period_is(self):
        audio='<主体1>说道{你好,,朋友}'
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',audio)
        client=FakeSyncClient(semantic_reply(issues=[semantic_issue('自然指令','{你好,,朋友}','规范标点','标点符号硬伤')],
                                             corrected_sections={'自然指令':audio.replace(',,','，')+'。'}))
        report=SafeAutoFixAgent(client=client).check(raw)
        self.assertIn(audio+'。',report['revised_caption'])
        self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')
        self.assertFalse(next(i for i in report['issues'] if i['source']=='llm')['resolved'])

    def test_model_grammar_and_unknown_typos_cannot_expand_whitelist(self):
        for kind,old,new in [('语法病句与语句不通顺','向前跑','快速奔向前方'),
                             ('无歧义错别字','白袍','长袍')]:
            raw=GOOD
            field='自然指令' if kind.startswith('语法') else '主体'
            client=FakeSyncClient(semantic_reply(issues=[semantic_issue(field,old,new,kind)],
                                                 corrected_sections={field:qa.parse_caption_sections(raw)[field].replace(old,new).strip()}))
            report=SafeAutoFixAgent(client=client).check(raw)
            self.assertEqual(report['revised_caption'],raw)
            self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')

    def test_model_failure_never_becomes_pass(self):
        report=SafeAutoFixAgent(client=FakeSyncClient(RuntimeError('synthetic failure')),max_retries=0).check(GOOD)
        self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')
        self.assertFalse(report['semantic_qa_performed'])

    def test_auto_decisions_survive_report_roundtrip_and_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'data.jsonl';cache=root/'result.jsonl'
            source.write_text(json.dumps({'id':'0007','caption':GOOD.replace('<主体1>向前跑。','男子向前跑')})+'\n')
            asyncio.run(qa.aprocess_batch(source,cache,SafeAutoFixAgent(dry_run=True),column='caption'))
            doc=make_review(cache,source,source,caption_column='caption',id_column='id')
            self.assertEqual(doc['workflow_version'],AUTO_WORKFLOW)
            self.assertEqual(summary(doc)['automatic_issues'],1)
            self.assertEqual(summary(doc)['pending'],1)
            self.assertEqual(next(iter(approved_changes(doc).values()))[0]['mode'],'automatic_rule')
            rows=[[i['issue_id'],i['sample_id'],i['field'],i['label'],'\n'.join(i['excerpts']),i['subject'],
                   '已自动修正' if i['decision']=='auto' else '待审核',''] for i in doc['issues']]
            target=root/'review.csv'
            def write():
                with target.open('w',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(rows)
            write();imported=import_review(doc,target,revision=0)
            self.assertEqual(summary(imported)['automatic_issues'],1)
            next(row for row in rows if row[6]=='待审核')[6]='已自动修正';write()
            with self.assertRaisesRegex(ValueError,'不能将其他问题'):
                import_review(doc,target,revision=0)

    def test_default_web_exports_copy_automatically_and_restores_without_reapplying(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);app=web.App(root/'out',root/'jobs')
            self.addCleanup(app.close)
            raw=GOOD.replace('<主体1>向前跑。','男子向前跑')
            data=(json.dumps({'id':'0007','caption':raw,'note':'原样保留'},ensure_ascii=False)+'\n').encode()
            with patch('caption_qa_review_report.export_review',return_value={}),patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
                item=app.create_job('auto.jsonl',io.BytesIO(data),len(data),dry_run=True)
                app.executor.submit(lambda:None).result(timeout=10)
            job=app.jobs[item['id']];snapshot=job.snapshot()
            self.assertEqual(snapshot['workflow_version'],AUTO_WORKFLOW)
            self.assertEqual(job.status,'done',job.error)
            result=snapshot['review_output']
            self.assertEqual(result['remaining_review_items'],1)
            corrected=Path(result['corrected_path'])
            output=json.loads(corrected.read_text())
            self.assertEqual(output['caption'],raw.replace('男子向前跑','男子向前跑。'))
            self.assertEqual(output['id'],'0007');self.assertEqual(output['note'],'原样保留')
            self.assertEqual(job.source.read_bytes(),data)
            app.save_history();before=set((root/'out').iterdir())
            restored=web.App(root/'out',root/'jobs')
            try:
                self.assertEqual(restored.jobs[job.id].snapshot()['review_output'],result)
                self.assertEqual(before,set((root/'out').iterdir()))
            finally:restored.close()


if __name__=='__main__':unittest.main()
