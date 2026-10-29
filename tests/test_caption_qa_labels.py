"""Metadata separator fixes, review migration and final exported values."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from captionqa import caption_qa_agent as qa, caption_qa_web as web
from captionqa.caption_qa_labels import separator_operations, normalize_separators, enrich_separator_fixes, POLICY
from captionqa.caption_qa_review import SafeAutoFixAgent, IdentificationAgent, _review_records, summary, approved_changes
from captionqa.caption_qa_review_store import ReviewStore
from tests.test_caption_qa_upgrades import GOOD

RAW='音色-相同，环境音-同类不同个'
FIXED='音色-相同｜环境音-同类不同个'


def document(value=RAW,automatic=True):
    text=GOOD+'## audio_similarity_tag\n'+value+'\n'
    agent=SafeAutoFixAgent(dry_run=True) if automatic else IdentificationAgent(dry_run=True)
    report=agent.check(text)
    return _review_records({1:{'id':'0001','caption':text,'qa_report':report}},caption_column='caption',id_column='id',source_sha256=qa._source_sha256(Path(__file__)))


class LabelTests(unittest.TestCase):
    def test_clear_separators_and_legacy_aliases_preserve_label_contents_and_whitespace(self):
        cases={'任务类型':('人物参考','动态参考'),'task_type':('人物参考','动态参考'),
               'subtask_type':('人物参考','动态参考'),'audio_similarity_tag':('音色-相同','环境音-同类不同个'),
               'video_similarity_tag':('实体/IP - 相同','镜头/运镜-同类不同个'),
               'similarity_tag':('实体/ip-相同','剪辑方式-不同'),'视听标签':('实体/IP-相同','音乐-相同')}
        for field,(left,right) in cases.items():
            for delimiter in ('，',',','、','；',';','\n','\r\n'):
                original='\n  '+left+delimiter+' '+right+'\n\n'
                with self.subTest(field=field,delimiter=delimiter):
                    self.assertEqual(normalize_separators(field,original),'\n  '+left+'｜ '+right+'\n\n')
                    operations=separator_operations(field,original)
                    self.assertEqual(len(operations),1)
                    self.assertEqual(original[operations[0]['start']:operations[0]['end']],delimiter)

    def test_legal_pipe_width_and_internal_slashes_stay_unchanged(self):
        for text in ('音色-相同|环境音-同类不同个','音色-相同｜环境音-同类不同个'):
            self.assertEqual(normalize_separators('audio_similarity_tag',text),text)
        self.assertEqual(normalize_separators('video_similarity_tag','实体/IP-相同|场景-相同，镜头/运镜-相同'),
                         '实体/IP-相同|场景-相同|镜头/运镜-相同')

    def test_uncertain_or_empty_labels_and_audio_prose_never_auto_change(self):
        for text in ('音色-相同，','，音乐-相同','音色-相同||环境音-相同','音色-相同，\n环境音-相同',
                     '音色-相同\n\n环境音-相同','无，音色-相同','未知维度-相同，音乐-相同',
                     '音色-相同，环境中传来风声','音色-相同，环境音-大概相同',
                     '音色-相同。\n环境音-同类不同个','音色-相同\n## 未知标题\n环境音-相同'):
            self.assertEqual(separator_operations('audio_similarity_tag',text),[],text)
        for field in ('提示','自然指令','输入描述','画面元素','主体','similarity_label'):
            self.assertEqual(normalize_separators(field,RAW),RAW)

    def test_safe_auto_and_review_all_follow_selected_workflow(self):
        raw=GOOD+'## audio_similarity_tag\n'+RAW+'\n'
        report=SafeAutoFixAgent(dry_run=True).check(raw)
        self.assertEqual(report['status'],'AUTO_FIXED')
        self.assertIn(FIXED,report['revised_caption'])
        self.assertTrue(next(i for i in report['issues'] if i['issue_type']=='label_separator')['resolved'])
        review=document();self.assertEqual(summary(review)['human_issues'],0)
        self.assertEqual(summary(review)['automatic_issues'],1)
        self.assertIn(FIXED,approved_changes(review)[1][0]['after'])
        manual=document(automatic=False)
        self.assertEqual(summary(manual)['needs_confirmation'],1)
        self.assertEqual(approved_changes(manual),{})

    def test_saved_pending_findings_upgrade_once_and_keep_human_decisions(self):
        for decision in ('keep','keep_highlight','reject','approve','manual'):
            review=document();issue=review['issues'][0]
            issue['decision']=decision;issue['operations']=[];issue.pop('separator_policy',None)
            before=copy.deepcopy(review)
            self.assertEqual(enrich_separator_fixes(review),[])
            self.assertEqual(review,before)
        review=document();issue=review['issues'][0];issue['decision']='pending';issue['operations']=[];issue.pop('separator_policy',None)
        self.assertEqual(enrich_separator_fixes(review),[issue['issue_id']])
        self.assertEqual(issue['decision'],'auto')
        self.assertEqual(enrich_separator_fixes(review),[])
        review=document();issue=review['issues'][0];issue['decision']='pending';issue['operations']=[];issue.pop('separator_policy',None)
        review['field_edits'][issue['field_id']]=RAW
        self.assertEqual(enrich_separator_fixes(review),[])

    def test_export_refuses_tampered_separator_edit(self):
        review=document();issue=review['issues'][0]
        issue['operations'][0]['new']='｜音乐-不同｜'
        with self.assertRaisesRegex(ValueError,'分隔符'):approved_changes(review)


class LabelExportTests(unittest.TestCase):
    def test_old_saved_review_upgrades_without_api_and_csv_contains_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);app=web.App(root/'out',root/'jobs');self.addCleanup(app.close)
            text=GOOD+'## audio_similarity_tag\n'+RAW+'\n'
            stream=io.StringIO(newline='');csv.writer(stream).writerows([['id','caption','备注'],['0001',text,'保持原样']])
            raw=stream.getvalue().encode();item=app.create_job('sample.csv',io.BytesIO(raw),len(raw),dry_run=True)
            app.executor.submit(lambda:None).result(timeout=20)
            job=app.jobs[item['id']];store=app._review_store(job);review=store.load()
            label=next(i for i in review['issues'] if i['kind']=='label_separator')
            label.update(decision='pending',operations=[],can_auto_fix=False);label.pop('separator_policy',None)
            review['events']=[];store.save(review);revision=review['revision']
            source_before=job.source.read_bytes();cache_before=job.cache.read_bytes()
            migrated=store.load()
            self.assertEqual(migrated['revision'],revision+1)
            self.assertEqual(summary(migrated)['human_issues'],0)
            self.assertEqual(migrated['events'][-1]['policy'],POLICY)
            self.assertEqual(store.load(),migrated)
            plan=store.plan(migrated);self.assertFalse(plan.blockers)
            self.assertIn(FIXED,plan.captions[1])
            output=app.apply_review(job.id,migrated['revision'])
            with Path(output['corrected_path']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['id'],'0001')
            self.assertEqual(rows[0]['备注'],'保持原样')
            self.assertIn(FIXED,rows[0]['caption'])
            self.assertEqual(job.source.read_bytes(),source_before)
            self.assertEqual(job.cache.read_bytes(),cache_before)


if __name__=='__main__':unittest.main()
