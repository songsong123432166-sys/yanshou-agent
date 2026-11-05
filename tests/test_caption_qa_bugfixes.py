"""User-facing regressions: retries, legacy headings, keeps, final draft checks."""
import asyncio
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from captionqa import caption_qa_agent as qa,caption_qa_web as web
from captionqa.caption_field_schema import ALIASES,FIELDS,normalize,render
from captionqa.caption_qa_labels import pipe_width_operations
from captionqa.caption_qa_review import SafeAutoFixAgent,IdentificationAgent,_review_records
from captionqa.caption_qa_review_io import export_approved
from captionqa.caption_qa_review_model import AUTO_WORKFLOW
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_resume import prepare_retry
from captionqa.caption_qa_review_state import decide,summary
from captionqa.caption_qa_review_store import ReviewStore,source_records
from tests.test_caption_qa_upgrades import GOOD,FakeSyncClient,semantic_reply


class LegacyAliasFixTests(unittest.TestCase):
    def test_english_summary_cannot_hide_undefined_subject(self):
        raw=GOOD.replace('<主体1>奔跑。','<主体99>奔跑。').replace('## 总结','## summary')
        result=SafeAutoFixAgent(dry_run=True).check(raw)
        self.assertEqual(result['status'],'NEED_HUMAN_REVIEW')
        self.assertEqual(result['subject_check']['ghost_subjects'],['<主体99>'])
        self.assertNotIn('summary',qa.parse_caption_sections(raw))
        self.assertTrue(qa.instruction_parser_outdated(raw,5))
        self.assertFalse(qa.instruction_parser_outdated(raw,6))

    def test_english_scene_duplicate_and_tone_are_in_real_check_scope(self):
        raw=GOOD+'## scene\n<场景1>：房间。\n<场景1>：走廊。\n## tone\n写实。\n## visual_element\n无。\n'
        client=FakeSyncClient(semantic_reply())
        result=qa.CaptionQAAgent(client=client,max_retries=0).check(raw)
        self.assertIn('duplicate_scene_numbers',{i['issue_type'] for i in result['issues']})
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertIn('场景',payload['caption_sections']);self.assertIn('基调',payload['caption_sections'])
        self.assertIn('画面元素',payload['read_only_sections'])

    def test_shared_schema_converts_all_four_aliases_without_ai(self):
        fields={n:'无。' for n in FIELDS};fields['任务类型']='人物参考'
        expected=render(fields);raw=expected
        for alias,title in {'tone':'基调','scene':'场景','summary':'总结','visual_element':'画面元素'}.items():
            self.assertEqual(ALIASES[alias],title);raw=raw.replace('## '+title+'\n','## '+alias+'\n')
        result=asyncio.run(normalize(raw))
        self.assertEqual(result['caption'],expected);self.assertEqual(result['model_requests'],0)

    def test_populated_chinese_scene_wins_over_old_alias_without_mixing(self):
        raw=GOOD+'## scene\n<场景9>：旧房间。\n## 场景\n<场景1>：当前房间。\n'
        parsed=qa.parse_caption_sections(raw)
        self.assertEqual(parsed['场景'],'<场景1>：当前房间。\n')
        self.assertTrue(parsed.field_resolution_notes)


class KeptFormatFixTests(unittest.TestCase):
    def build(self,raw):
        record={'id':'keep-001','caption':raw,'qa_report':IdentificationAgent(dry_run=True).check(raw)}
        doc=_review_records({1:record},caption_column='caption',id_column='id',source_sha256='synthetic')
        doc['workflow_version']=AUTO_WORKFLOW
        return record,doc

    def test_pending_and_explicit_keeps_survive_automatic_projection(self):
        raw=GOOD+'## 听觉标签\n音色 - 相同 ｜ 环境音 - 相同\n'
        record,base=self.build(raw)
        issue=next(i for i in base['issues'] if i['kind']=='pipe_width')
        for choice in ['pending','defer','keep','reject','keep_highlight']:
            with self.subTest(choice=choice):
                doc=decide(base,revision=0,issue_id=issue['issue_id'],decision=choice)
                snapshot=copy.deepcopy(doc);plan=compile_review(doc,{1:record}).require_valid()
                self.assertEqual(plan.rows[0].final_text,raw)
                self.assertEqual(len(plan.highlights),int(choice=='keep_highlight'))
                self.assertEqual(doc,snapshot)
        approved=decide(base,revision=0,issue_id=issue['issue_id'],decision='approve')
        self.assertIn('音色-相同|环境音-相同',compile_review(approved,{1:record}).rows[0].final_text)

    def test_kept_case_and_width_do_not_cancel_an_independent_period_fix(self):
        raw=GOOD.replace('<主体1>向前跑。','<主体1>向前跑')+'## 视觉标签\n实体/IP - 相同 ｜ 场景 - 相同\n'
        record,doc=self.build(raw)
        for issue in list(doc['issues']):
            doc=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],
                decision='keep_highlight' if issue['kind'] in {'pipe_width','tag_language'} else 'approve')
        plan=compile_review(doc,{1:record}).require_valid()
        self.assertIn('实体/IP - 相同 ｜ 场景 - 相同',plan.rows[0].final_text)
        self.assertIn('<主体1>向前跑。',plan.rows[0].final_text)

    def test_preserved_width_still_allows_confirmed_comma_separator_change(self):
        raw=GOOD+'## 听觉标签\n音色 - 相同，环境音 - 相同\n'
        record,doc=self.build(raw)
        field=next(f for f in doc['fields'] if f['name']=='audio_similarity_tag')
        label=next(i for i in doc['issues'] if i['kind']=='label_separator')
        kept=copy.deepcopy(label);kept.update(issue_id='synthetic-width-keep',kind='pipe_width',decision='keep',
            operations=pipe_width_operations(field['name'],field['original']))
        doc['issues'].append(kept);field['issue_ids'].append(kept['issue_id'])
        doc=decide(doc,revision=0,issue_id=label['issue_id'],decision='approve')
        final=compile_review(doc,{1:record}).require_valid().rows[0].final_text
        self.assertIn('音色 - 相同|环境音 - 相同',final)


class RetryAndDraftFixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'input.jsonl';self.cache=self.root/'checkpoint.jsonl'
        self.store=ReviewStore(self.source,self.source,self.cache,'caption','id')

    def build(self,raw=GOOD,automatic=True):
        self.source.write_text(json.dumps({'id':'sample-001','caption':raw,'note':'untouched'},ensure_ascii=False)+'\n')
        agent=(SafeAutoFixAgent if automatic else IdentificationAgent)(dry_run=True)
        asyncio.run(qa.aprocess_batch(self.source,self.cache,agent,column='caption'))
        return self.store.load()

    def append_new_result(self,clear_findings=False):
        rows=[json.loads(s) for s in self.cache.read_text().splitlines()];row=copy.deepcopy(rows[-1])
        if clear_findings:
            row['qa_report']['issues']=[];row['qa_report']['static_errors']=[]
            row['qa_report']['need_human_review']=False
        with self.cache.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')

    def test_saved_manual_field_survives_changed_checkpoint_and_new_pass(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>向前跑'))
        field=next(f for f in doc['fields'] if f['name']=='自然指令')
        draft='0-4秒，<主体1>先转头，再向前跑。'
        doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text=draft);self.store.save(doc)
        old_bytes=self.store.path.read_bytes();original=self.source.read_bytes()
        prepare_retry(self.store);self.append_new_result(clear_findings=True)
        restored=self.store.load();before=copy.deepcopy(restored)
        self.assertEqual(restored['field_edits'],doc['field_edits'])
        self.assertGreater(restored['revision'],doc['revision'])
        self.assertEqual(self.store.load(),restored)
        self.assertFalse(self.cache.with_suffix('.review_resume.json').exists())
        self.assertTrue(any(p.read_bytes()==old_bytes for p in self.root.glob('*.review_previous_*.json')))
        plan=self.store.plan(restored).require_valid()
        output=export_approved(restored,self.source,self.source,self.cache,self.root/'out','sample',plan=plan)
        row=json.loads(Path(output['corrected_path']).read_text())
        self.assertIn(draft,row['caption']);self.assertEqual(row['note'],'untouched')
        self.assertEqual(self.source.read_bytes(),original);self.assertEqual(restored,before)

    def test_approvals_keeps_highlights_and_defer_survive_retry(self):
        raw=GOOD.replace('<主体1>向前跑。','主体1向前跑')+'## 听觉标签\n未知内容，待核对\n'
        doc=self.build(raw,automatic=False)
        for issue in list(doc['issues']):
            choice='approve' if issue['kind']=='missing_punctuation' else 'keep_highlight' if issue['kind']=='bare_tags' else 'defer'
            doc=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],decision=choice)
        self.store.save(doc);expected={i['issue_id']:i['decision'] for i in doc['issues']}
        prepare_retry(self.store);self.append_new_result();restored=self.store.load()
        self.assertEqual({i['issue_id']:i['decision'] for i in restored['issues']},expected)
        plan=self.store.plan(restored).require_valid();self.assertEqual(len(plan.highlights),1)
        self.assertIn('主体1向前跑。',plan.rows[0].final_text)
        self.assertNotIn('<主体1>向前跑',plan.rows[0].final_text)

    def test_retry_report_order_changes_keep_the_confirmed_occurrence_and_unique_ids(self):
        raw=GOOD.replace('<主体1>向前跑。','主体1看向主体1。')
        doc=self.build(raw,automatic=False)
        bare=[i for i in doc['issues'] if i['kind']=='bare_tags'];self.assertEqual(len(bare),2)
        selected=bare[1]
        doc=decide(doc,revision=doc['revision'],issue_id=selected['issue_id'],decision='approve')
        self.store.save(doc);prepare_retry(self.store)
        row=json.loads(self.cache.read_text().splitlines()[-1])
        row['qa_report']['issues'].insert(0,{'source':'llm','section':'自然指令','original_snippet':'看向',
            'issue_type':'语法病句与语句不通顺','suggestion':'合成的待核对描述。','can_auto_fix':False,'resolved':False})
        with self.cache.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        restored=self.store.load();ids=[i['issue_id'] for i in restored['issues']]
        self.assertEqual(len(ids),len(set(ids)))
        retained=next(i for i in restored['issues'] if i['issue_id']==selected['issue_id'])
        self.assertEqual(retained['decision'],'approve');self.assertEqual(retained['operations'],selected['operations'])
        final=self.store.plan(restored).require_valid().rows[0].final_text
        self.assertIn('主体1看向<主体1>。',final)
        self.assertNotIn('<主体1>看向<主体1>',final)

    def test_retry_preparation_is_idempotent_and_pending_marker_survives_app_restart(self):
        doc=self.build(GOOD.replace('白袍。','白袍'))
        field=doc['fields'][0];doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text='<主体1>：白袍男子。')
        self.store.save(doc);prepare_retry(self.store)
        marker=self.cache.with_suffix('.review_resume.json');data=marker.read_bytes()
        prepare_retry(self.store);self.assertEqual(marker.read_bytes(),data)
        self.append_new_result()
        reopened=ReviewStore(self.source,self.source,self.cache,'caption','id').load()
        self.assertEqual(reopened['field_edits'],doc['field_edits'])
        self.assertEqual(sum(e.get('decision')=='retry_review_resumed' for e in reopened['events']),1)

    def test_failed_recovery_save_retains_original_review_and_can_be_retried(self):
        doc=self.build(GOOD.replace('白袍。','白袍'))
        field=doc['fields'][0];doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text='<主体1>：白袍男子。')
        self.store.save(doc);prepare_retry(self.store);self.append_new_result()
        original=self.store.path.read_bytes()
        with patch.object(ReviewStore,'save',side_effect=OSError('synthetic disk error')):
            with self.assertRaises(OSError):self.store.load()
        self.assertEqual(self.store.path.read_bytes(),original)
        self.assertTrue(self.cache.with_suffix('.review_resume.json').exists())
        self.assertEqual(self.store.load()['field_edits'],doc['field_edits'])

    def test_stale_cleanup_marker_does_not_restore_an_older_manual_draft(self):
        doc=self.build(GOOD.replace('白袍。','白袍'))
        field=doc['fields'][0];doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text='<主体1>：白袍男子。')
        self.store.save(doc);prepare_retry(self.store)
        marker=self.cache.with_suffix('.review_resume.json');old_marker=marker.read_bytes()
        self.append_new_result();current=self.store.load()
        current=decide(current,revision=current['revision'],field_id=field['field_id'],final_text='<主体1>：身穿白袍的青年。')
        self.store.save(current);marker.write_bytes(old_marker);self.append_new_result()
        restored=self.store.load()
        self.assertEqual(restored['field_edits'],current['field_edits'])
        self.assertFalse(marker.exists())

    def test_source_change_does_not_apply_old_manual_text_or_delete_recovery(self):
        doc=self.build(GOOD.replace('白袍。','白袍'))
        field=doc['fields'][0];doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text='<主体1>：白袍男子。')
        self.store.save(doc);prepare_retry(self.store);old=self.store.path.read_bytes()
        self.source.write_text(self.source.read_text().replace('sample-001','different-id'))
        with self.assertRaisesRegex(ValueError,'来源文件发生变化'):self.store.load()
        self.assertEqual(self.store.path.read_bytes(),old)
        self.assertTrue(self.cache.with_suffix('.review_resume.json').exists())

    def test_manual_new_undefined_reference_warns_and_export_remains_available(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>向前跑'))
        field=next(f for f in doc['fields'] if f['name']=='自然指令')
        final='0-4秒，<主体1>走到<主体99>身边。'
        doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text=final)
        plan=self.store.plan(doc).require_valid()
        self.assertEqual(summary(doc)['pending'],0)
        warning=next(w for w in plan.review_checks if w.get('kind')=='ghost_subjects')
        self.assertEqual(warning['sample_id'],'sample-001');self.assertEqual(warning['excerpt'],'<主体99>')
        self.assertEqual(plan.public()['review_check_fields'],1)
        output=export_approved(doc,self.source,self.source,self.cache,self.root/'out','sample',plan=plan)
        self.assertEqual(output['post_review_fields'],1)
        self.assertIn(final,json.loads(Path(output['corrected_path']).read_text())['caption'])

    def test_manually_introduced_subject_and_scene_duplicates_warn(self):
        raw=GOOD+'## 场景\n<场景1>：室内。\n'
        for field_name,draft,kind in [('主体','<主体1>：男子。\n<主体1>：女子。','duplicate_defined_subjects'),
            ('场景','<场景1>：室内。\n<场景1>：走廊。','duplicate_scene_numbers')]:
            with self.subTest(field_name=field_name):
                record={'id':'sample-001','caption':raw,'qa_report':SafeAutoFixAgent(dry_run=True).check(raw)}
                doc=_review_records({1:record},caption_column='caption',id_column='id',source_sha256='synthetic')
                fid='synthetic-'+field_name
                doc['fields'].append({'field_id':fid,'row_index':1,'sample_id':'sample-001','name':field_name,
                    'original':qa.parse_caption_sections(raw)[field_name],'issue_ids':[],
                    'source_sheet':None,'source_row':None})
                doc['field_edits'][fid]=draft
                plan=compile_review(doc,{1:record}).require_valid()
                self.assertTrue(any(w.get('kind')==kind and w['field_id']==fid for w in plan.review_checks))

    def test_real_web_retry_keeps_manual_draft_in_export(self):
        app=web.App(self.root/'out',self.root/'jobs');self.addCleanup(app.close)
        raw=GOOD.replace('## 总结','<主体2>：女子。\n## 总结')
        data=(json.dumps({'id':'web-001','caption':raw})+'\n').encode()
        created=app.create_job('synthetic.jsonl',io.BytesIO(data),len(data),dry_run=True,column='caption')
        app.executor.submit(lambda:None).result(timeout=10);job=app.jobs[created['id']]
        payload=app.review(job.id);field=next(f for f in payload['review']['fields'] if f['name']=='自然指令')
        draft=field['original'].strip()+'<主体2>站立。'
        payload=app.review(job.id,{'revision':payload['review']['revision'],'field_id':field['field_id'],'final_text':draft})
        source=job.source.read_bytes();checkpoint=job.cache.read_bytes();job.status='partial'
        app.retry(job.id);app.executor.submit(lambda:None).result(timeout=10)
        self.assertEqual(job.status,'done',job.error)
        payload=app.review(job.id)
        self.assertEqual(len(payload['review']['field_edits']),1)
        output=app.apply_review(job.id,payload['review']['revision'])
        self.assertIn(draft,json.loads(Path(output['corrected_path']).read_text())['caption'])
        self.assertEqual(job.source.read_bytes(),source);self.assertEqual(job.cache.read_bytes(),checkpoint)


if __name__=='__main__':unittest.main()
