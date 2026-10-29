"""Shared final-text contract across UI payloads, review decisions and exporters."""
import asyncio
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import caption_qa_agent as qa
from caption_qa_review import IdentificationAgent, SafeAutoFixAgent, make_review, decide, approved_changes
from caption_qa_review_io import export_approved
from caption_qa_review_plan import compile_review, document_fingerprint
from caption_qa_review_store import ReviewStore, source_records
from caption_qa_review_model import DECISION_LABELS
from test_caption_qa_upgrades import GOOD
import test_caption_qa_review as fixtures

RAW=GOOD.replace('<主体1>向前跑。','主体1向前跑')


class ReviewProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source.csv';self.cache=self.root/'checkpoint.jsonl'

    def build(self, caption=RAW, automatic=False):
        with self.source.open('w',encoding='utf-8-sig',newline='') as f:
            csv.writer(f).writerows([['id','caption','note'],['0001',caption,'untouched']])
        agent=(SafeAutoFixAgent if automatic else IdentificationAgent)(dry_run=True)
        asyncio.run(qa.aprocess_batch(self.source,self.cache,agent,column='caption'))
        return make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def choose(self,doc,kind,decision):
        issue=next(i for i in doc['issues'] if i['kind']==kind)
        return decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],decision=decision)

    def plan(self,doc):return compile_review(doc,source_records(doc,self.source))

    def export(self,doc,plan):
        return export_approved(doc,self.source,self.source,self.cache,self.root/'out','data',plan=plan)

    def test_one_final_field_drives_editor_caption_and_export_without_mutating_review(self):
        doc=self.choose(self.build(),'missing_punctuation','approve')
        doc=self.choose(doc,'bare_tags','keep_highlight');before=copy.deepcopy(doc)
        plan=self.plan(doc).require_valid();field=next(f for f in plan.fields if f.name=='自然指令')
        self.assertEqual(field.final_text.strip(),'0-4秒，第0秒，主体1向前跑。')
        self.assertEqual(field.editor_text,field.final_text.strip())
        self.assertEqual(plan.changes,approved_changes(doc));self.assertEqual(plan.captions[1],RAW.replace('主体1向前跑','主体1向前跑。'))
        result=self.export(doc,plan)
        with Path(result['corrected_path']).open(encoding='utf-8-sig',newline='') as f:row=list(csv.reader(f))[1]
        self.assertEqual(qa.parse_caption_sections(row[1])['自然指令'],plan.public()['fields'][field.field_id]['final_text'])
        self.assertEqual(result['plan_fingerprint'],plan.fingerprint)
        self.assertEqual(row[2],'untouched');self.assertEqual(doc,before)

    def test_consumers_cannot_mutate_the_final_plan_through_public_metadata(self):
        doc=self.choose(self.build(),'missing_punctuation','approve')
        doc=self.choose(doc,'bare_tags','keep_highlight');plan=self.plan(doc).require_valid()
        changes=plan.changes;changes[1][0]['after']='unexpected mutation'
        plan.highlights[1]['issue_ids'].clear()
        plan.audit[0]['修改后']='unexpected mutation'
        public=plan.public();public['fields'][plan.fields[0].field_id]['final_text']='unexpected mutation'
        info=self.export(doc,plan)
        with Path(info['corrected_path']).open(encoding='utf-8-sig',newline='') as f:row=list(csv.reader(f))[1]
        self.assertIn('主体1向前跑。',row[1]);self.assertNotIn('unexpected mutation',row[1])
        self.assertTrue(plan.highlights[1]['issue_ids'])

    def test_manual_draft_overrides_automatic_text_but_keeps_its_provenance_hidden(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','男子向前跑'),automatic=True)
        auto_ids={i['issue_id'] for i in doc['issues'] if i['decision']=='auto'}
        doc=self.choose(doc,'missing_subjects','keep_highlight');field=doc['fields'][0]
        doc=decide(doc,revision=doc['revision'],field_id=field['field_id'],final_text='男子向前走。')
        plan=self.plan(doc).require_valid();result=plan.fields[0]
        self.assertEqual(result.editor_text,'男子向前走。');self.assertEqual(result.mode,'human')
        self.assertTrue(auto_ids.issubset(set(plan.public()['automatic_issue_ids'])))
        self.assertTrue(auto_ids.isdisjoint(set(plan.public()['visible_issue_ids'])))
        self.assertEqual(len(plan.highlights),1);self.assertIn('男子向前走。',plan.captions[1])

    def test_invalid_saved_draft_remains_editable_and_is_located_before_export(self):
        doc=self.build();field=doc['fields'][0];bad='原文。\n## 新字段\n不允许新增。'
        doc=decide(doc,revision=0,field_id=field['field_id'],final_text=bad)
        plan=self.plan(doc);payload=plan.public()
        self.assertFalse(payload['export_ready']);self.assertEqual(payload['blockers'][0]['field_id'],field['field_id'])
        self.assertEqual(payload['fields'][field['field_id']]['editor_text'],bad)
        self.assertIn('标题或边界',payload['blockers'][0]['message'])
        with self.assertRaisesRegex(ValueError,'题目 0001'):self.export(doc,plan)
        self.assertFalse((self.root/'out').exists())
        doc=decide(doc,revision=1,field_id=field['field_id'],final_text='0-4秒，主体1向前跑。')
        fixed=self.plan(doc).require_valid();self.assertTrue(fixed.public()['export_ready'])
        self.assertTrue(Path(self.export(doc,fixed)['corrected_path']).is_file())

    def test_conflicting_operations_are_visible_blockers_and_never_partial_edits(self):
        doc=self.build();field=doc['fields'][0]
        for n,issue in enumerate(doc['issues']):
            issue['decision']='approve';issue['operations']=[{'start':0,'end':1,'old':field['original'][:1],'new':str(n+1)}]
        plan=self.plan(doc)
        self.assertFalse(plan.public()['export_ready']);self.assertIn('冲突',plan.blockers[0]['message'])
        self.assertEqual(plan.fields[0].final_text,field['original'])
        with self.assertRaisesRegex(ValueError,'冲突'):self.export(doc,plan)

    def test_same_revision_but_different_decisions_cannot_reuse_a_stale_plan(self):
        doc=self.choose(self.build(),'missing_punctuation','approve');plan=self.plan(doc)
        changed=copy.deepcopy(doc)
        next(i for i in changed['issues'] if i['kind']=='missing_punctuation')['decision']='keep'
        self.assertEqual(changed['revision'],doc['revision'])
        self.assertNotEqual(document_fingerprint(changed),plan.fingerprint)
        with self.assertRaisesRegex(ValueError,'审核预览已变化'):self.export(changed,plan)
        self.assertFalse((self.root/'out').exists())

    def test_changed_source_field_is_reported_with_its_identity(self):
        doc=self.choose(self.build(),'missing_punctuation','approve')
        records=source_records(doc,self.source);records[1]['caption']=GOOD
        plan=compile_review(doc,records)
        self.assertFalse(plan.public()['export_ready'])
        self.assertEqual(plan.blockers[0]['sample_id'],'0001');self.assertEqual(plan.blockers[0]['field'],'自然指令')
        self.assertIn('原文已变化',plan.blockers[0]['message'])

    def test_summary_cache_failure_does_not_lose_a_successfully_saved_review(self):
        import caption_qa_review_store as storage
        doc=self.build();store=ReviewStore(self.source,self.source,self.cache,'caption','id');store.save(doc)
        updated=self.choose(doc,'missing_punctuation','approve');write=storage.atomic_json
        def fail_summary(path,value):
            if str(path).endswith('.review_summary.json'):raise OSError('synthetic derived cache failure')
            write(path,value)
        with patch.object(storage,'atomic_json',side_effect=fail_summary):store.save(updated)
        self.assertEqual(json.loads(self.cache.with_suffix('.review_summary.json').read_text())['revision'],0)
        self.assertEqual(storage.read_review_summary(self.cache)['revision'],1)
        self.assertEqual(store.load()['revision'],1)
        self.assertEqual(store.payload(store.load())['projection']['fingerprint'],self.plan(updated).fingerprint)

    def test_legacy_json_store_roundtrip_preserves_decisions_and_projection(self):
        doc=self.choose(self.build(),'bare_tags','keep_highlight')
        doc=self.choose(doc,'missing_punctuation','approve')
        store=ReviewStore(self.source,self.source,self.cache,'caption','id')
        store.save(doc);loaded=store.load()
        expected=store.payload(doc);actual=store.payload(loaded)
        self.assertEqual(actual['projection'],expected['projection'])
        self.assertEqual(actual['summary'],expected['summary'])
        self.assertEqual(actual['projection']['decision_labels'],DECISION_LABELS)


class ReviewProjectionWebTests(unittest.TestCase):
    setUp=fixtures.ReviewWebTests.setUp
    request=fixtures.ReviewWebTests.request
    job=fixtures.ReviewWebTests.job

    def test_preview_and_export_fingerprints_match_and_stale_previews_are_rejected(self):
        identifier=self.job();initial=self.app.review(identifier);issue=initial['review']['issues'][0]
        updated=self.app.review(identifier,{'revision':0,'issue_id':issue['issue_id'],'decision':'approve'})
        self.assertNotEqual(initial['projection']['fingerprint'],updated['projection']['fingerprint'])
        route=f'/api/jobs/{identifier}/review/apply'
        status,_=self.request('POST',route,body=json.dumps({'revision':1,'projection_fingerprint':initial['projection']['fingerprint']}).encode())
        self.assertEqual(status,400)
        status,body=self.request('POST',route,body=json.dumps({'revision':1,'projection_fingerprint':updated['projection']['fingerprint']}).encode())
        self.assertEqual(status,200,body);result=json.loads(body)
        self.assertEqual(result['plan_fingerprint'],updated['projection']['fingerprint'])
        row=json.loads(Path(result['corrected_path']).read_text())
        field=updated['review']['fields'][0]
        self.assertEqual(qa.parse_caption_sections(row['caption'])[field['name']],updated['projection']['fields'][field['field_id']]['final_text'])

    def test_job_status_uses_saved_review_when_the_summary_cache_is_stale(self):
        import caption_qa_review_store as storage
        identifier=self.job();payload=self.app.review(identifier);issue=payload['review']['issues'][0]
        write=storage.atomic_json
        def fail_summary(path,value):
            if str(path).endswith('.review_summary.json'):raise OSError('synthetic derived cache failure')
            write(path,value)
        with patch.object(storage,'atomic_json',side_effect=fail_summary):
            updated=self.app.review(identifier,{'revision':0,'issue_id':issue['issue_id'],'decision':'approve'})
        self.assertEqual(updated['summary']['pending'],0)
        self.assertEqual(self.app.jobs[identifier].snapshot()['review_summary']['pending'],0)
        self.assertEqual(self.app.jobs[identifier].snapshot()['review_summary']['revision'],1)

    def test_invalid_manual_save_returns_a_repairable_projection_and_reset_clears_it(self):
        identifier=self.job();initial=self.app.review(identifier);field=initial['review']['fields'][0]
        result=self.app.review(identifier,{'revision':0,'field_id':field['field_id'],'final_text':'正文。\n## 未知\n正文。'})
        self.assertFalse(result['projection']['export_ready']);self.assertEqual(result['projection']['blockers'][0]['field_id'],field['field_id'])
        reset=self.app.review(identifier,{'revision':1,'field_id':field['field_id'],'reset':True})
        self.assertTrue(reset['projection']['export_ready']);self.assertFalse(reset['review']['field_edits'])

if __name__=='__main__':unittest.main()
