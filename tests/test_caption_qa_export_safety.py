"""Regressions for omissions reported during real human-review export audits."""
import asyncio
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import captionqa.caption_qa_agent as qa
from captionqa.caption_qa_candidates import candidate_operations, raw_candidate_operations
from captionqa.caption_qa_candidate_context import contextual_operations, normalize_saved_candidates
from captionqa.caption_qa_review import IdentificationAgent, make_review, decide
from captionqa.caption_qa_review_edits import resolve_fields
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_store import source_records, ReviewStore
from captionqa.caption_qa_review_io import export_approved
from tests.test_caption_qa_upgrades import GOOD


def apply(text,ops):
    for o in sorted(ops,key=lambda o:(o['start'],o['end']),reverse=True):
        text=text[:o['start']]+o['new']+text[o['end']:]
    return text


def suggestion(snippet,new,kind='标点符号硬伤'):
    return dict(kind=kind,field='自然指令',snippet=snippet,reason=f'改为“{new}”。',
        suggestion_candidate=dict(start=0,source=snippet,replacement=new,explanation='明确修改'))


class CandidateContextTests(unittest.TestCase):
    def test_existing_outer_period_is_not_appended_again(self):
        old='{你好。}';text=old+'。';issue=suggestion(old,old+'。')
        raw=raw_candidate_operations(issue,text)
        self.assertTrue(raw)
        self.assertEqual(candidate_operations(issue,text),[])
        self.assertEqual(apply(text,candidate_operations(issue,text)),text)

    def test_timecode_wrapper_move_keeps_other_edits_and_preview_matches_export(self):
        old='<第0秒，伴随奔跑脚步声>';text=old+'。'
        issue=suggestion(old,'第0秒，<伴随奔跑脚步声>。','内容符号不规范')
        ops=candidate_operations(issue,text)
        self.assertTrue(ops)
        self.assertEqual(apply(text,ops),'第0秒，<伴随奔跑脚步声>。')
        self.assertTrue(all(o['candidate_new']=='第0秒，<伴随奔跑脚步声>' for o in ops))

    def test_combining_comma_replacement_with_static_append_does_not_double_period(self):
        text='牌匾【停云轩】，'
        ops=[dict(start=len(text)-1,end=len(text),old='，',new='。'),
             dict(start=len(text),end=len(text),old='',new='。')]
        self.assertEqual(apply(text,contextual_operations(text,ops)),'牌匾【停云轩】。')
        # Do not omit an insertion if the adjacent original mark is deleted.
        text='片段。';ops=[dict(start=2,end=2,old='',new='。'),dict(start=2,end=3,old='。',new='')]
        self.assertEqual(apply(text,contextual_operations(text,ops)),'片段。')

    def test_audio_screen_and_expressive_ascii_are_protected(self):
        for text,char in [('{好。}','。'),('【好。】','。'),('片段...','.'),('片段！！','！')]:
            p=text.index(char)
            ops=[dict(start=p,end=p,old='',new=char)]
            self.assertEqual(contextual_operations(text,ops),ops)


class SavedExportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source.csv';self.cache=self.root/'checkpoint.jsonl'

    def build(self,caption):
        self.cache.unlink(missing_ok=True)
        with self.source.open('w',encoding='utf-8-sig',newline='') as f:
            csv.writer(f).writerows([['id','caption','note'],['0001',caption,'保留其他列']])
        asyncio.run(qa.aprocess_batch(self.source,self.cache,IdentificationAgent(dry_run=True),column='caption'))
        return make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def plan(self,doc):return compile_review(doc,source_records(doc,self.source))

    def semantic(self,doc,field,snippet,kind='无歧义错别字'):
        doc=copy.deepcopy(doc);f=next(f for f in doc['fields'] if f['name']==field)
        old=next(i for i in doc['issues'] if i['field_id']==f['field_id'])
        old.update(kind=kind,origin='llm',snippet=snippet,operations=[],reason='请核对。')
        return doc,f,old

    def test_manual_unchanged_and_original_semantic_snippet_are_warnings_not_export_blocks(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>看着树腾。'))
        # Add an original static finding so this field belongs to the review.
        if not doc['fields']:
            doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>看着树腾'))
        doc,f,issue=self.semantic(doc,'自然指令','树腾')
        doc=decide(doc,revision=doc['revision'],field_id=f['field_id'],final_text=f['original'].strip())
        before=copy.deepcopy(doc);plan=self.plan(doc)
        self.assertTrue(plan.public()['export_ready']);self.assertTrue(plan.review_checks)
        self.assertEqual(doc,before)
        out=export_approved(doc,self.source,self.source,self.cache,self.root/'out','data',plan=plan)
        self.assertEqual(out['post_review_fields'],1)
        self.assertTrue(json.loads(Path(out['recheck_path']).read_text())['post_review_checks'])
        kept=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],decision='keep_highlight')
        self.assertEqual(kept['field_edits'],doc['field_edits'])
        self.assertFalse(self.plan(kept).review_checks)

    def test_cross_field_undefined_reference_is_checked_after_manual_summary_change(self):
        caption=GOOD.replace('<主体1>奔跑。','<主体7>奔跑。').replace('<主体1>向前跑。','<主体1>看着<主体7>。')
        doc=self.build(caption);issue=next(i for i in doc['issues'] if i['kind']=='ghost_subjects')
        f=next(f for f in doc['fields'] if f['field_id']==issue['field_id'])
        doc=decide(doc,revision=doc['revision'],field_id=f['field_id'],final_text='<主体1>奔跑。')
        self.assertTrue(any(w.get('issue_id')==issue['issue_id'] for w in self.plan(doc).review_checks))
        # Resolve via the definition field; the reference itself need not change.
        doc=self.build(caption.replace('身穿白袍。','身穿白袍'))
        f=next(f for f in doc['fields'] if f['name']=='主体')
        doc=decide(doc,revision=doc['revision'],field_id=f['field_id'],final_text='<主体1>：身穿白袍。\n<主体7>：青年。')
        for i in doc['issues']:
            if i['kind']=='ghost_subjects':i['decision']='manual'
        self.assertFalse(any(w.get('issue_id')==i['issue_id'] for w in self.plan(doc).review_checks for i in doc['issues'] if i['kind']=='ghost_subjects'))

    def test_manual_text_is_never_cleaned_silently(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>向前跑'))
        f=doc['fields'][0];final='0-4秒，<主体1>向前跑。。'
        doc=decide(doc,revision=0,field_id=f['field_id'],final_text=final)
        plan=self.plan(doc)
        self.assertEqual(plan.fields[0].editor_text,final)
        self.assertTrue(any('连续标点' in w['message'] for w in plan.review_checks))
        self.assertTrue(plan.public()['export_ready'])

    def test_fixing_another_issue_does_not_hide_an_unchanged_implicit_subject_snippet(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','一名男子向前跑'))
        doc,f,issue=self.semantic(doc,'自然指令','一名男子向前跑','隐式漏挂主体标签')
        doc=decide(doc,revision=0,field_id=f['field_id'],final_text=f['original'].strip().replace('第0秒','第1秒')+'。')
        self.assertTrue(any(w.get('issue_id')==issue['issue_id'] for w in self.plan(doc).review_checks))

    def test_punctuation_inserted_beside_a_semantic_anchor_is_not_a_missed_edit(self):
        doc=self.build(GOOD.replace('<主体1>向前跑。','<主体1>向前跑随后停下'))
        doc,f,issue=self.semantic(doc,'自然指令','<主体1>向前跑','语法病句与语句不通顺')
        final=f['original'].strip().replace('<主体1>向前跑','<主体1>向前跑，')+'。'
        doc=decide(doc,revision=0,field_id=f['field_id'],final_text=final)
        self.assertFalse(self.plan(doc).review_checks)

    def test_saved_candidates_upgrade_idempotently_without_changing_decisions_or_manual_draft(self):
        old='<第0秒，伴随奔跑脚步声>';text=old+'。'
        doc=self.build(GOOD.replace('<主体1>向前跑。',text))
        f=next(f for f in doc['fields'] if f['name']=='自然指令')
        issue=doc['issues'][0];issue.update(suggestion(old,'第0秒，<伴随奔跑脚步声>。','内容符号不规范'))
        issue['suggestion_candidate']['start']=f['original'].index(old)
        issue.update(operations=raw_candidate_operations(issue,f['original']),requires_confirmation=True,decision='approve',candidate_policy='confirmed-suggestion-v1')
        doc['issues']=[issue];f['issue_ids']=[issue['issue_id']];doc['fields']=[f]
        before=copy.deepcopy(doc)
        # Resolver accepts a historical, reproducible candidate without mutating it.
        self.assertNotIn('。。',resolve_fields(doc)[0].final_text);self.assertEqual(doc,before)
        store=ReviewStore(self.source,self.source,self.cache);store.save(doc)
        updated=store.load();self.assertEqual(updated['revision'],1)
        self.assertEqual(updated['issues'][0]['decision'],'approve')
        self.assertEqual(store.load(),updated)
        self.assertEqual(normalize_saved_candidates(updated),[])
        doc['field_edits'][f['field_id']]='原人工稿。。'
        normalize_saved_candidates(doc)
        self.assertEqual(resolve_fields(doc)[0].editor_text,'原人工稿。。')


if __name__=='__main__':unittest.main()
