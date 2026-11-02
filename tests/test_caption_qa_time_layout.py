"""Ranges form paragraphs; point events and quoted/audio content stay intact."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from captionqa import caption_qa_agent as qa
from captionqa.caption_qa_time_layout import normalize,operations,enrich_time_layout
from captionqa.caption_qa_review import SafeAutoFixAgent,IdentificationAgent,make_review,decide
from captionqa.caption_qa_review_store import ReviewStore,source_records
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_io import export_approved
from tests.test_caption_qa_upgrades import GOOD

TEXT='0-2秒，<主体1>奔跑。第1秒，转头。2-8秒，<主体1>停下。第4秒，微笑。'
EXPECTED=TEXT.replace('。2-8秒','。\n\n2-8秒')


class TimeLayoutTests(unittest.TestCase):
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
        for text in ['说明从0-2秒开始，随后继续。','0-2秒的画面是静止的。','第2秒，镜头切换。第4秒，转头。']:
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
