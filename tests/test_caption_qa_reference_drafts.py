"""Interactive recommendations can be uncertain and contextual, never automatic."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from captionqa import caption_qa_agent as qa, caption_qa_web as web
from captionqa.caption_qa_review import IdentificationAgent, make_review, decide, approved_changes
from captionqa.caption_qa_candidates import candidate_operations
from captionqa.caption_qa_suggestions import prepare_proposal
from captionqa.caption_qa_reference_drafts import POLICY
from tests.test_caption_qa_upgrades import GOOD, FakeAsyncClient, FakeSyncClient, semantic_reply, semantic_issue

OLD='0-8秒，镜头后，移全景跟拍'
NEW='0-8秒，镜头后移，全景跟拍'
REASON='逗号位置导致断句不通，疑为“镜头后移，全景跟拍”；涉及断句，需人工确认。'


class ReferenceDraftTests(unittest.TestCase):
    def draft(self,original,source,target,*,snippet=OLD,kind='语法病句与语句不通顺',ready=True,sections=None,assumptions=None):
        issue=dict(kind=kind,field='自然指令',reason=REASON,snippet=snippet)
        field=dict(name='自然指令',original=original)
        client=FakeSyncClient(dict(ready=ready,source=source,replacement=target,explanation='给出最小改动的推荐表达。',assumptions=assumptions or []))
        proposal=prepare_proposal(issue,field,api_key='fake',base_url='offline',model='fake',client=client,context_sections=sections)
        proposed=dict(issue,suggestion_candidate=proposal)
        ops=candidate_operations(proposed,original)
        final=original
        for op in reversed(ops):final=final[:op['start']]+op['new']+final[op['end']:]
        return proposal,ops,final,client

    def test_screenshot_clause_has_a_concrete_recommendation_even_when_report_says_manual(self):
        text=OLD+'<主体1>面带笑容向前奔跑。'
        proposal,ops,final,client=self.draft(text,OLD,NEW)
        self.assertTrue(ops);self.assertEqual(final,text.replace(OLD,NEW))
        self.assertEqual(proposal['policy'],POLICY)
        self.assertTrue(all(o['review_only'] for o in ops))
        self.assertIn('需要人工确认不构成拒绝生成的理由',client.calls[0]['messages'][0]['content'])

    def test_short_highlight_is_a_locator_not_a_rewrite_boundary(self):
        text=OLD+'<主体1>奔跑。'
        _,ops,final,_=self.draft(text,'镜头后，移全景跟拍','镜头后移，全景跟拍',snippet='镜头后')
        self.assertTrue(ops);self.assertEqual(final,text.replace(OLD,NEW))

    def test_uncertain_but_concrete_answer_is_retained_with_check_notes(self):
        _,ops,final,_=self.draft(OLD,OLD,NEW,ready=False,assumptions=['按镜头后移理解，请核对视频。'])
        self.assertTrue(ops);self.assertEqual(final,NEW)
        p,_,_,_=self.draft(OLD,OLD,NEW,ready=False)
        self.assertTrue(p['assumptions'])

    def test_manual_review_only_response_can_use_the_reports_existing_concrete_answer(self):
        issue=dict(kind='语法病句与语句不通顺',field='自然指令',snippet=OLD,
            reason='“镜头后，移全景跟拍”断句不通，疑为“镜头后移，全景跟拍”；需人工确认。',
            replacement=NEW+'<主体1>')  # stale detector payload must not win
        p=prepare_proposal(issue,dict(name='自然指令',original=OLD+'<主体1>奔跑。'),api_key='fake',
            base_url='offline',model='fake',client=FakeSyncClient({'ready':False,'explanation':'需要人工确认镜头运动。'}))
        self.assertEqual(p['source'],'镜头后，移全景跟拍')
        self.assertEqual(p['replacement'],'镜头后移，全景跟拍')
        self.assertTrue(p['assumptions'])

    def test_actual_field_context_can_support_a_defined_reference_replacement(self):
        text='0-8秒，<主体7>奔跑。'
        context={'主体':'<主体1>：青年。','自然指令':text}
        proposal,ops,final,client=self.draft(text,'<主体7>','<主体1>',snippet='<主体7>',kind='ghost_subjects',sections=context)
        self.assertTrue(ops);self.assertEqual(final,'0-8秒，<主体1>奔跑。')
        self.assertIn('<主体1>',proposal['known_reference_tags'])
        sent=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertEqual(sent['context_sections']['主体'],context['主体'])
        with self.assertRaises(ValueError):self.draft(text,'<主体7>','<主体99>',snippet='<主体7>',kind='ghost_subjects',sections=context)

    def test_other_paragraphs_times_numbers_and_movement_cannot_change(self):
        text=OLD+'<主体1>向前奔跑。\n第9秒，镜头切换。'
        bad=[text.replace('第9秒','第10秒'),text.replace('镜头切换','镜头结束'),
             text.replace('向前','向后'),text.replace('0-8秒','0-6秒')]
        for target in bad:
            with self.assertRaises(ValueError):self.draft(text,text,target)

    def test_audio_and_screen_words_are_preserved_but_punctuation_can_be_recommended(self):
        for source,target in [('{Now，go}','{Now,go}'),('【你好，朋友】','【你好,朋友】')]:
            _,ops,final,_=self.draft(source,source,target,snippet=source,kind='标点符号硬伤')
            self.assertTrue(ops);self.assertEqual(final,target)
        for source,target in [('{你好}','{再见}'),('【LAUNDRY】','【LAUNDRIES】'),('<发出脚步声>','<发出枪声>')]:
            with self.assertRaises(ValueError):self.draft(source,source,target,snippet=source)

    def test_a_manual_review_message_or_new_heading_is_not_a_draft(self):
        for target in ['需要人工审核','## 主体\n<主体1>：青年。','人工核对后再修改。']:
            with self.assertRaises(ValueError):self.draft(OLD,OLD,target)

    def test_disabled_checks_stay_disabled(self):
        with self.assertRaisesRegex(ValueError,'停用'):
            self.draft('红衣变蓝衣','红衣变蓝衣','红衣不变',snippet='红衣变蓝衣',kind='属性冲突')

    def test_generate_then_confirm_then_export_without_changing_the_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.jsonl';cache=root/'checkpoint.jsonl'
            caption=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',OLD+'<主体1>面带笑容向前奔跑。')
            source.write_text(json.dumps({'id':'demo-001','caption':caption,'note':'原样'})+'\n')
            raw=source.read_bytes()
            item=dict(semantic_issue('自然指令',OLD,REASON,'语法病句与语句不通顺'),can_auto_fix=False)
            asyncio.run(qa.aprocess_batch(source,cache,IdentificationAgent(async_client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True))),column='caption'))
            doc=make_review(cache,source,source,caption_column='caption',id_column='id')
            app=web.App(root/'out',root/'jobs');self.addCleanup(app.close)
            job=web.Job('a'*32,'synthetic.jsonl',source,root/'out/ids.txt',cache,False,'caption',None,8,status='done')
            job.selected_column='caption';job.id_column='id';app.jobs[job.id]=job;app._save_review(job,doc)
            issue=doc['issues'][0];before=copy.deepcopy(doc)
            client=FakeSyncClient(dict(ready=True,source=OLD,replacement=NEW,explanation='把逗号移到“移”后。',assumptions=['按镜头后移理解。']))
            original_prepare=prepare_proposal
            with patch.object(web,'read_saved_key',return_value='fake'),patch('captionqa.caption_qa_suggestions.prepare_proposal',side_effect=lambda i,f,**kw:original_prepare(i,f,client=client,**kw)):
                result=app.propose_review(job.id,doc['revision'],issue['issue_id'])
            draft=result['review'];self.assertEqual(draft['issues'][0]['decision'],'pending')
            self.assertEqual(approved_changes(draft),{});self.assertEqual(source.read_bytes(),raw)
            rejected=decide(draft,revision=draft['revision'],issue_id=issue['issue_id'],decision='keep')
            self.assertEqual(approved_changes(rejected),{})
            approved=decide(draft,revision=draft['revision'],issue_id=issue['issue_id'],decision='approve')
            app._save_review(job,approved);output=app.apply_review(job.id,approved['revision'])
            exported=json.loads(Path(output['corrected_path']).read_text())
            self.assertEqual(exported['caption'],caption.replace(OLD,NEW));self.assertEqual(exported['note'],'原样')
            self.assertEqual(source.read_bytes(),raw);self.assertEqual(doc,before)


if __name__=='__main__':unittest.main()
