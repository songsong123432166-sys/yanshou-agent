"""Cut thresholds, point line breaks, protected text and approved export parity."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from captionqa import caption_qa_agent as qa
from captionqa.caption_qa_time_layout import normalize,operations,enrich_time_layout,legacy_operations,previous_operations
from captionqa.caption_qa_review import SafeAutoFixAgent,IdentificationAgent,make_review,decide
from captionqa.caption_qa_review_store import ReviewStore,source_records
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_io import export_approved
from tests.test_caption_qa_upgrades import GOOD

TEXT='0-2秒，<主体1>奔跑。第1秒，转头。2-8秒，<主体1>停下。第4秒，微笑。'
EXPECTED=TEXT.replace('。2-8秒','。\n\n2-8秒')


class TimeLayoutTests(unittest.TestCase):
    def test_only_cut_events_start_lines_and_other_timestamps_stay_inline(self):
        text='0-8秒，<主体1>站立。第1秒，抬头。第2秒，镜头切换。第3秒，挥手。第4秒，镜头切换。第5秒，微笑。'
        expected=text.replace('。第2秒','。\n第2秒').replace('。第4秒','。\n第4秒')
        self.assertEqual(normalize('自然指令',text),expected)
        self.assertEqual(normalize('自然指令',expected),expected)
        plain='0-8秒，<主体1>站立。镜头切换，特写。第2秒，转头。'
        self.assertEqual(normalize('自然指令',plain),plain.replace('。镜头切换','。\n镜头切换'))

    def test_three_cuts_add_one_marker_but_two_cuts_split_point_events(self):
        text='0-8秒，<主体1>站立。第2秒，镜头切换，转头。第4秒，镜头切至侧面。第6秒，切镜，微笑。'
        expected=text.replace('0-8秒，','0-8秒，镜头多次切换，')
        self.assertEqual(normalize('自然指令',text),expected)
        self.assertEqual(normalize('自然指令',expected),expected)
        low=text.replace('第6秒，切镜','第6秒，镜头后拉')
        self.assertEqual(normalize('自然指令',low),low.replace('。第2秒','。\n第2秒').replace('。第4秒','。\n第4秒'))

    def test_cut_counts_reset_at_each_range_and_high_count_does_not_merge_lines(self):
        high='0-8秒，<主体1>站立。\n第2秒，镜头切换。第4秒，镜头切换。第6秒，镜头切换。'
        low='8-12秒，<主体1>转头。第9秒，镜头切换。第11秒，镜头切换。'
        expected=high.replace('0-8秒，','0-8秒，镜头多次切换，')+'\n\n'+low.replace('。第9秒','。\n第9秒').replace('。第11秒','。\n第11秒')
        self.assertEqual(normalize('自然指令',high+low),expected)

    def test_cut_keywords_in_protected_text_and_negative_statements_do_not_count(self):
        literals=['{第2秒，镜头切换。第4秒，镜头切换。第6秒，镜头切换}',
            '（背景音乐：镜头切换、切镜、镜头切换）','<传来切镜、切镜、切镜的声音>',
            '【第2秒，镜头切换。镜头切换。镜头切换】','“镜头切换。镜头切换。镜头切换”']
        for literal in literals:
            text='0-8秒，<主体1>展示'+literal+'。第7秒，转头。'
            self.assertEqual(normalize('自然指令',text),text)
        text='0-8秒，<主体1>站立，不切镜，不切镜，不切镜。第2秒，镜头上摇。第4秒，镜头前推。'
        self.assertEqual(normalize('自然指令',text),text)

    def test_opening_cut_is_a_boundary_not_an_extra_cut_within_the_range(self):
        text='0-8秒，镜头切换，<主体1>站立。第2秒，镜头切换。第4秒，镜头切换。'
        self.assertEqual(normalize('自然指令',text),text.replace('。第2秒','。\n第2秒').replace('。第4秒','。\n第4秒'))

    def test_declared_multi_cut_preserves_point_layout_and_crlf_low_count_is_supported(self):
        text='0-8秒，镜头多次切换，<主体1>站立。第2秒，转头。\n第4秒，微笑。'
        self.assertEqual(normalize('自然指令',text),text)
        low='0 - 8秒，<主体1>站立。第2秒，转头。\r\n第4秒，微笑。'
        self.assertEqual(normalize('自然指令',low),low)

    def test_ranges_and_point_events_have_different_layout(self):
        self.assertEqual(normalize('自然指令',TEXT),EXPECTED)
        self.assertIn('。第1秒，转头。',EXPECTED);self.assertIn('。第4秒，微笑。',EXPECTED)
        self.assertEqual(normalize('自然指令',EXPECTED),EXPECTED)
        self.assertEqual(''.join(EXPECTED.split()),''.join(TEXT.split()))

    def test_single_newline_becomes_one_blank_line_existing_blank_lines_remain(self):
        text=TEXT.replace('。2-8秒','。\n2-8秒')
        self.assertEqual(normalize('自然指令',text),EXPECTED)
        self.assertEqual(normalize('自然指令',EXPECTED.replace('\n\n','\n\n\n')),EXPECTED.replace('\n\n','\n\n\n'))
        self.assertEqual(normalize('自然指令',text.replace('\n','\r\n')),EXPECTED.replace('\n','\r\n'))

    def test_decimal_ranges_and_seconds_units_are_recognized(self):
        for heading in ['2.5-8.5秒','2～8秒','2秒至8秒','2-8s','第2-8秒']:
            text='0-2秒，<主体1>站立。'+heading+'，<主体1>转头。'
            self.assertEqual(normalize('自然指令',text),text.replace('。'+heading,'。\n\n'+heading))

    def test_metadata_definitions_and_regular_mentions_are_not_split(self):
        for field in ['提示','输入描述','主体','任务类型']:
            self.assertEqual(normalize(field,TEXT),TEXT)
        for text in ['说明从0-2秒开始，随后继续。','0-2秒的画面是静止的。']:
            self.assertEqual(normalize('自然指令',text),text)

    def test_audio_screen_quotes_and_technical_literals_are_opaque(self):
        for text in ['<主体1>说道{0-2秒，开始。2-8秒，继续}。',
                     '（背景音乐：0-2秒，渐强。2-8秒，继续）。',
                     '<声音从0-2秒，持续到2-8秒，停止>。',
                     '显示【0-2秒，开始。2-8秒，结束】。',
                     '原文是“0-2秒，开始。2-8秒，结束”。',
                     '网址 https://example.test/2-8秒，原样保留。']:
            self.assertEqual(normalize('自然指令',text),text)

    def test_rule_does_not_check_order_and_does_not_force_point_events_into_new_paragraphs(self):
        text='4-8秒，<主体1>站立。0-2秒，<主体1>转头。\n第1秒，微笑。'
        result=normalize('自然指令',text)
        self.assertEqual(result,text.replace('。0-2秒','。\n\n0-2秒'))


class TimeLayoutExportTests(unittest.TestCase):
    def test_old_v2_point_breaks_do_not_reintroduce_the_superseded_rule(self):
        doc=self.build();field=doc['fields'][0];old=copy.deepcopy(doc['issues'][0])
        old.update(issue_id='previous-v2',decision='auto',operations=previous_operations('自然指令',field['original']))
        doc['issues']=[old];field['issue_ids']=[old['issue_id']]
        before=copy.deepcopy(doc);plan=compile_review(doc,source_records(doc,self.source)).require_valid()
        self.assertIn(EXPECTED,plan.rows[0].final_text)
        self.assertNotIn('。\n第1秒',plan.rows[0].final_text)
        self.assertEqual(doc,before)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input.jsonl';self.cache=self.root/'checkpoint.jsonl'
        self.caption=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',TEXT)

    def build(self,automatic=True):
        self.source.write_text(json.dumps({'id':'0001','caption':self.caption,'note':'原样'})+'\n')
        cls=SafeAutoFixAgent if automatic else IdentificationAgent
        asyncio.run(qa.aprocess_batch(self.source,self.cache,cls(dry_run=True),column='caption'))
        return make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def test_default_pipeline_automatically_exports_the_layout_and_hides_the_issue(self):
        doc=self.build();plan=compile_review(doc,source_records(doc,self.source));payload=plan.public()
        self.assertFalse(payload['visible_issue_ids']);self.assertEqual(len(payload['automatic_issue_ids']),1)
        result=export_approved(doc,self.source,self.source,self.cache,self.root/'out','sample',plan=plan)
        exported=json.loads(Path(result['corrected_path']).read_text())
        self.assertEqual(exported['caption'],self.caption.replace(TEXT,EXPECTED))
        self.assertEqual(exported['note'],'原样');self.assertEqual(json.loads(self.source.read_text())['caption'],self.caption)

    def test_same_range_inside_speech_does_not_steal_the_real_heading_location(self):
        text='0-2秒，<主体1>说道{2-8秒，继续}。2-8秒，<主体1>停下。'
        report=SafeAutoFixAgent(dry_run=True).check(GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text))
        self.assertIn(text.replace('。2-8秒','。\n\n2-8秒'),report['revised_caption'])
        self.assertTrue(next(i for i in report['issues'] if i['issue_type']=='time_range_break')['resolved'])

    def test_kept_point_layout_is_not_applied_when_other_layout_is_approved(self):
        self.caption=self.caption.replace('第1秒，转头','第1秒，镜头切换').replace('第4秒，微笑','第4秒，镜头切换')
        doc=self.build(False)
        for issue in list(doc['issues']):
            doc=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],
                       decision='keep' if issue['snippet']=='第1秒' else 'approve')
        plan=compile_review(doc,source_records(doc,self.source)).require_valid()
        self.assertIn('。第1秒',plan.rows[0].final_text)
        self.assertIn('。\n第4秒',plan.rows[0].final_text)
        self.assertIn('\n\n2-8秒',plan.rows[0].final_text)

    def test_legacy_range_approval_remains_valid_without_approving_new_point_breaks(self):
        doc=self.build(False);field=doc['fields'][0];original=field['original']
        legacy=copy.deepcopy(doc['issues'][0]);legacy.update(issue_id='legacy',decision='approve',operations=legacy_operations('自然指令',original))
        doc['issues']=[legacy];field['issue_ids']=['legacy']
        before=copy.deepcopy(doc)
        plan=compile_review(doc,source_records(doc,self.source)).require_valid()
        self.assertIn(TEXT.replace('。2-8秒','。\n\n2-8秒'),plan.rows[0].final_text)
        self.assertEqual(doc,before)
        enrich_time_layout(doc,source_records(doc,self.source))
        self.assertEqual(sum(i['decision']=='pending' for i in doc['issues']),0)

    def test_multi_cut_marker_composes_with_a_confirmed_sentence_edit_and_exports(self):
        text='0-8秒，<主体1>奔跑。第2秒，镜头切换。第4秒，镜头切换。第6秒，镜头切换。'
        self.caption=self.caption.replace(TEXT,text)
        doc=self.build();field=doc['fields'][0];original=field['original']
        rewrite=copy.deepcopy(doc['issues'][0]);rewrite.update(issue_id='rewrite',kind='语法病句与语句不通顺',origin='llm',decision='approve',
            operations=[dict(start=0,end=len(original),old=original,new=original.replace('奔跑','转头'))])
        doc['issues'].append(rewrite);field['issue_ids'].append('rewrite')
        plan=compile_review(doc,source_records(doc,self.source)).require_valid()
        expected=text.replace('0-8秒，','0-8秒，镜头多次切换，').replace('奔跑','转头')
        self.assertIn(expected,plan.rows[0].final_text)
        result=export_approved(doc,self.source,self.source,self.cache,self.root/'out','sample',plan=plan)
        self.assertEqual(json.loads(Path(result['corrected_path']).read_text())['caption'],plan.rows[0].final_text)

    def test_review_all_mode_requires_approval_and_keep_is_respected(self):
        doc=self.build(False);issue=next(i for i in doc['issues'] if i['kind']=='time_range_break')
        self.assertEqual(issue['decision'],'pending')
        doc=decide(doc,revision=0,issue_id=issue['issue_id'],decision='keep')
        self.assertEqual(enrich_time_layout(doc,source_records(doc,self.source)),[])
        self.assertFalse(compile_review(doc,source_records(doc,self.source)).captions)

    def test_layout_composes_after_an_overlapping_approved_sentence_rewrite(self):
        doc=self.build();field=doc['fields'][0];original=field['original']
        issue=copy.deepcopy(doc['issues'][0]);issue.update(issue_id='rewrite',kind='语法病句与语句不通顺',origin='llm',decision='approve',
            operations=[dict(start=0,end=len(original),old=original,new=original.replace('微笑','点头'))])
        doc['issues'].append(issue);field['issue_ids'].append(issue['issue_id'])
        plan=compile_review(doc,source_records(doc,self.source)).require_valid()
        self.assertIn(EXPECTED.replace('微笑','点头'),plan.rows[0].final_text)

    def test_old_reviews_upgrade_once_without_replacing_human_decisions_or_manual_text(self):
        doc=self.build();doc['issues']=[];doc['fields']=[];doc['events']=[]
        store=ReviewStore(self.source,self.source,self.cache);store.save(doc)
        upgraded=store.load();self.assertEqual(upgraded['revision'],1)
        self.assertEqual(upgraded['issues'][0]['decision'],'auto')
        self.assertEqual(store.load(),upgraded)
        field=upgraded['fields'][0]
        manual=decide(upgraded,revision=1,field_id=field['field_id'],final_text=TEXT)
        before=copy.deepcopy(manual)
        self.assertFalse(enrich_time_layout(manual,source_records(manual,self.source)))
        self.assertEqual(manual,before)
        self.assertEqual(compile_review(manual,source_records(manual,self.source)).fields[0].editor_text,TEXT)

    def test_first_open_of_old_checkpoint_already_has_the_rule_without_a_stale_retry(self):
        self.build()
        row=json.loads(self.cache.read_text());report=row['qa_report']
        report.update(issues=[],static_errors=[],remaining_static_errors=[],need_human_review=False,
                      proposed_caption=self.caption,revised_caption=self.caption)
        row['qa_revised_caption']=self.caption
        self.cache.write_text(json.dumps(row)+'\n')
        store=ReviewStore(self.source,self.source,self.cache,column='caption',id_column='id')
        first=store.load()
        self.assertEqual(first['revision'],0)
        self.assertEqual(first['issues'][0]['decision'],'auto')
        second=store.load()
        self.assertEqual(second['revision'],first['revision'])
        self.assertIn(EXPECTED,store.plan(first).rows[0].final_text)


if __name__=='__main__':unittest.main()
