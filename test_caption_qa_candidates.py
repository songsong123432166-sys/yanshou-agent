"""AI can propose a precise unknown typo; only human approval executes it."""
import asyncio
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_candidates import candidate_operation, candidate_operations, enrich_candidates
from caption_qa_review import SafeAutoFixAgent, IdentificationAgent, make_review, approved_changes, decide, summary
from caption_qa_review_io import export_approved
from caption_qa_review_report import HEADERS, import_review
from test_caption_qa_upgrades import GOOD, FakeSyncClient, FakeAsyncClient, semantic_reply, semantic_issue

REASON='疑似应为“树藤缝隙”（与主体中“树藤编的囚笼”一致），但“树腾”未命中确定性纠错词表，保留原文并转人工确认。'
RAW=GOOD.replace('<主体1>向前跑。','<主体1>挤在树腾缝隙后向外窥视。')


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input.jsonl';self.cache=self.root/'cache.jsonl'

    def build(self,structured=False):
        self.source.write_text(json.dumps({'id':'cm11247d98','caption':RAW,'note':'保持其他列'})+'\n')
        item=dict(semantic_issue('自然指令','树腾缝隙',REASON,'无歧义错别字'),can_auto_fix=False)
        if structured:item['replacement']='树藤缝隙'
        client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True,human_review_reason='需要确认候选错字'))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,SafeAutoFixAgent(async_client=client),column='caption'))
        self.assertEqual(len(client.calls),1)
        return make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def test_saved_example_and_legacy_phrasings_produce_one_local_character_change(self):
        for snippet,reason in [('树腾缝隙',REASON),
                               ('挤在树腾缝隙后','“树腾”疑为“树藤”（对应主体中的囚笼），需人工确认。'),
                               ('树腾','疑为"树藤"之误写，需人工确认后再改。')]:
            op=candidate_operation({'kind':'无歧义错别字','snippet':snippet,'reason':reason,'field':'自然指令'},'第5秒，<主体1>挤在树腾缝隙后向外窥视。')
            self.assertIsNotNone(op)
            self.assertEqual((op['old'],op['new']),('腾','藤'))
            self.assertTrue(op['review_only'])

    def test_default_detection_never_executes_candidate_without_confirmation(self):
        for structured in (False,True):
            self.cache.unlink(missing_ok=True)
            document=self.build(structured)
            cached=json.loads(self.cache.read_text())
            self.assertEqual(cached['qa_revised_caption'],RAW)
            self.assertEqual(cached['qa_status'],'NEED_HUMAN_REVIEW')
            self.assertEqual(summary(document)['automatic_issues'],0)
            self.assertEqual(approved_changes(document),{})
            issue=document['issues'][0]
            self.assertEqual(issue['decision'],'pending')
            self.assertTrue(issue['requires_confirmation'])
            self.assertTrue(issue['operations'])

    def test_confirmation_exports_only_that_typo_with_real_id_and_change_audit(self):
        document=self.build();issue=document['issues'][0]
        approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
        original=self.source.read_bytes()
        result=export_approved(approved,self.source,self.source,self.cache,self.root,'data')
        value=json.loads(Path(result['corrected_path']).read_text())
        self.assertEqual(value['caption'],RAW.replace('树腾','树藤'))
        self.assertEqual(value['id'],'cm11247d98');self.assertEqual(value['note'],'保持其他列')
        self.assertEqual(self.source.read_bytes(),original)
        audit=json.loads(Path(result['recheck_path']).read_text())
        self.assertEqual(audit['changes'][0]['修改来源'],'approved_candidate')
        self.assertEqual(audit['model_requests'],0)

    def test_reject_or_forged_automatic_decision_never_executes_candidate(self):
        document=self.build();issue=document['issues'][0]
        rejected=decide(document,revision=0,issue_id=issue['issue_id'],decision='reject')
        self.assertEqual(approved_changes(rejected),{})
        forged=copy.deepcopy(document);forged['issues'][0]['decision']='auto'
        with self.assertRaisesRegex(ValueError,'人工确认'):
            approved_changes(forged)
        approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
        approved['issues'][0]['operations'][0]['new']='根'
        with self.assertRaisesRegex(ValueError,'展示的指定改法'):
            approved_changes(approved)

    def test_audio_screen_facts_grammar_ambiguity_and_repeated_locations_are_not_executable(self):
        base={'kind':'无歧义错别字','snippet':'树腾','replacement':'树藤','reason':'疑为“树藤”','field':'自然指令'}
        for text in ['<主体1>说道{树腾}。','<主体1>举着【树腾】标牌。','（背景音乐是树腾）。','<传来树腾的声音>。','树腾与树腾。']:
            self.assertIsNone(candidate_operation(base,text),text)
        for old,new in [('白袍','红袍'),('左侧','右侧'),('三人','四人'),('Tom','Tim'),('树腾','<主体2>'),('树腾','树藤藤')]:
            self.assertIsNone(candidate_operation(dict(base,snippet=old,replacement=new),old))
        for reason in ['疑为“树藤”或“树根”。','不建议改为“树藤”。','“树藤”误写为“树腾”。']:
            self.assertIsNone(candidate_operation({k:v for k,v in dict(base,reason=reason).items() if k!='replacement'},'树腾'))
        self.assertIsNone(candidate_operation(dict(base,kind='语法病句与语句不通顺'),'树腾'))

    def test_optional_structured_candidate_is_validated_and_preserved_without_auto_merging(self):
        sections=qa.parse_caption_sections(RAW)
        item=dict(semantic_issue('自然指令','树腾缝隙',REASON,'无歧义错别字'),can_auto_fix=False,replacement='树藤缝隙')
        reply=semantic_reply(issues=[item],need_human_review=True)
        report=SafeAutoFixAgent(client=FakeSyncClient(reply)).check(RAW)
        self.assertEqual(report['issues'][0]['replacement'],'树藤缝隙')
        self.assertEqual(report['revised_caption'],RAW)
        reply['issues'][0]['replacement']=['bad']
        with self.assertRaisesRegex(ValueError,'候选替换'):
            qa._validate_semantic(reply,sections)

    def test_old_review_is_upgraded_once_without_model_request_or_overwriting_decisions(self):
        document=self.build()
        for issue in document['issues']:
            issue['operations']=[];issue.pop('requires_confirmation',None);issue.pop('candidate_policy',None)
        app=web.App(self.root/'out',self.root/'jobs');self.addCleanup(app.close)
        job=web.Job('a'*32,'data.jsonl',self.source,self.root/'out/ids.txt',self.cache,False,'caption',None,8,status='done')
        job.selected_column='caption';job.id_column='id';app.jobs[job.id]=job
        app._save_review(job,document)
        before=self.cache.read_bytes()
        upgraded=app.review(job.id)['review']
        self.assertEqual(upgraded['revision'],1)
        self.assertTrue(upgraded['issues'][0]['operations'])
        self.assertEqual(app.review(job.id)['review']['revision'],1)
        self.assertEqual(self.cache.read_bytes(),before)
        kept=copy.deepcopy(document);kept['issues'][0]['decision']='keep'
        self.assertEqual(enrich_candidates(kept),0)

    def test_csv_review_can_confirm_the_same_candidate_without_full_field_draft(self):
        document=self.build();issue=document['issues'][0]
        path=self.root/'review.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(HEADERS)
            writer.writerow([issue['issue_id'],issue['sample_id'],issue['field'],issue['label'],'\n'.join(issue['excerpts']),issue['subject'],'确认按建议改',''])
        imported=import_review(document,path,revision=0)
        self.assertEqual(imported['field_edits'],{})
        self.assertEqual(next(iter(approved_changes(imported).values()))[0]['after'].strip(),qa.parse_caption_sections(RAW.replace('树腾','树藤'))['自然指令'].strip())

    def test_old_music_brackets_and_voice_tag_colon_can_be_confirmed_without_changing_words(self):
        examples=[('<主体1>站立。(背景音乐加入打击乐的演奏，铜管演奏着紧张的主旋律)。',
                   '(背景音乐加入打击乐的演奏，铜管演奏着紧张的主旋律)',
                   '将半角括号“()”替换为全角“（）”，内部文字不变。'),
                  ('<主体1>:画外音，音色沉稳年轻，说普通话。','<主体1>:画外音',
                   '将半角冒号“:”替换为全角冒号“：”，与其他标签冒号保持一致。')]
        for text,snippet,reason in examples:
            self.cache.unlink(missing_ok=True)
            raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text)
            self.source.write_text(json.dumps({'id':'0007','caption':raw})+'\n')
            item=dict(semantic_issue('自然指令',snippet,reason,'标点符号硬伤'),can_auto_fix=False)
            client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True))
            asyncio.run(qa.aprocess_batch(self.source,self.cache,IdentificationAgent(async_client=client),column='caption'))
            document=make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')
            issue=next(i for i in document['issues'] if i['kind']=='标点符号硬伤')
            self.assertTrue(issue['requires_confirmation'])
            self.assertEqual(issue['decision'],'pending')
            self.assertEqual(approved_changes(document),{})
            approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
            result=export_approved(approved,self.source,self.source,self.cache,self.root,'punctuation')
            expected=raw.replace('(','（').replace(')','）') if '(' in text else raw.replace('<主体1>:','<主体1>：')
            self.assertEqual(json.loads(Path(result['corrected_path']).read_text())['caption'],expected)
            if len(issue['operations'])==2:
                corrupt=copy.deepcopy(approved);target=next(i for i in corrupt['issues'] if i['issue_id']==issue['issue_id'])
                target['operations'].pop()
                with self.assertRaisesRegex(ValueError,'完整执行'):
                    approved_changes(corrupt)

    def test_sound_description_width_candidate_needs_confirmation_and_preserves_audio_words(self):
        text='<主体1>挥刀，<发出挥刀的唰啦破空音效声,哒哒的脚步声和沉闷的肉体受击声>。'
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text)
        reason='将半角逗号替换为全角逗号：“唰啦破空音效声，哒哒的脚步声”'
        item=dict(semantic_issue('自然指令','唰啦破空音效声,哒哒的脚步声',reason,'标点符号硬伤'),can_auto_fix=True)
        client=FakeSyncClient(semantic_reply(issues=[item],corrected_sections={'自然指令':text.replace(',','，')}))
        report=SafeAutoFixAgent(client=client).check(raw)
        self.assertEqual(report['revised_caption'],raw)
        self.assertEqual(report['status'],'NEED_HUMAN_REVIEW')
        self.source.write_text(json.dumps({'id':'cm2f2036c6','caption':raw})+'\n')
        async_client=FakeAsyncClient(semantic_reply(issues=[item],corrected_sections={'自然指令':text.replace(',','，')}))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,SafeAutoFixAgent(async_client=async_client),column='caption'))
        document=make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')
        issue=next(i for i in document['issues'] if i['kind']=='标点符号硬伤')
        self.assertTrue(issue['operations'][0]['confirmed_audio_width'])
        self.assertEqual(approved_changes(document),{})
        approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
        result=export_approved(approved,self.source,self.source,self.cache,self.root,'sound')
        self.assertEqual(json.loads(Path(result['corrected_path']).read_text())['caption'],raw.replace(',','，'))
        for body in ['<主体1>说道{唰啦破空音效声,哒哒的脚步声}。','<主体1>举牌【唰啦破空音效声,哒哒的脚步声】。']:
            issue_copy=dict(issue,field='自然指令')
            self.assertEqual(candidate_operations(issue_copy,body),[])

    def test_explicit_dialogue_pair_preserves_words_and_uses_the_specified_direction(self):
        text='<主体1>说道{Now, if you\'ll excuse me，}。'
        reason='英语对白内部混用全角逗号（同一句对白前半段为半角“Now,”），建议将“me，}”改为“me,}”；因涉及对白原文保留，不自动修改，转人工确认。'
        issue={'kind':'标点符号硬伤','field':'自然指令','snippet':'me，}','reason':reason}
        ops=candidate_operations(issue,text)
        self.assertEqual(len(ops),1)
        self.assertEqual((ops[0]['old'],ops[0]['new']),('，',','))
        self.assertEqual((ops[0]['candidate_old'],ops[0]['candidate_new']),('me，}','me,}'))
        self.assertEqual(text[ops[0]['start']:ops[0]['end']],'，')
        self.assertTrue(ops[0]['review_only'])
        for old,new in [('me，}','me,}'),('真的，','真的？'),('你好。','你好！'),('哦，，','哦，'),('你好}','你好。}')]:
            value=f'<主体1>说道{{{old}。'
            structured=dict(issue,snippet=old,replacement=new,reason='请人工确认指定标点改法。')
            edits=candidate_operations(structured,value)
            self.assertTrue(edits,(old,new))
            for op in reversed(edits):value=value[:op['start']]+op['new']+value[op['end']:]
            self.assertEqual(value,f'<主体1>说道{{{new}。')

    def test_dialogue_candidate_is_not_automatic_but_confirmation_exports_exact_punctuation(self):
        text='<主体1>说道{Now, if you\'ll excuse me，}。'
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text)
        self.source.write_text(json.dumps({'id':'cma4fd67d4','caption':raw,'note':'其他列保留'})+'\n')
        reason='建议将“me，}”改为“me,}”；涉及对白原文，不自动修改，转人工确认。'
        item=dict(semantic_issue('自然指令','me，}',reason,'标点符号硬伤'),can_auto_fix=False)
        client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,SafeAutoFixAgent(async_client=client),column='caption'))
        self.assertEqual(json.loads(self.cache.read_text())['qa_revised_caption'],raw)
        document=make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')
        issue=next(i for i in document['issues'] if i['kind']=='标点符号硬伤')
        self.assertTrue(issue['requires_confirmation']);self.assertEqual(issue['decision'],'pending')
        self.assertEqual(approved_changes(document),{})
        approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
        original=self.source.read_bytes()
        result=export_approved(approved,self.source,self.source,self.cache,self.root,'dialogue')
        self.assertEqual(json.loads(Path(result['corrected_path']).read_text()),
                         {'id':'cma4fd67d4','caption':raw.replace('me，}','me,}'),'note':'其他列保留'})
        self.assertEqual(self.source.read_bytes(),original)
        rejected=decide(document,revision=0,issue_id=issue['issue_id'],decision='keep')
        self.assertEqual(approved_changes(rejected),{})
        forged=copy.deepcopy(approved);target=next(i for i in forged['issues'] if i['issue_id']==issue['issue_id'])
        target['decision']='auto'
        with self.assertRaisesRegex(ValueError,'人工确认'):approved_changes(forged)
        target['decision']='approve';target['operations'][0]['new']='！'
        with self.assertRaisesRegex(ValueError,'展示的指定改法'):approved_changes(forged)

    def test_explicit_punctuation_rejects_ambiguous_content_numeric_and_literal_changes(self):
        base={'kind':'标点符号硬伤','field':'自然指令','reason':'请人工确认建议。'}
        for old,new in [('你好，','您好，'),('hello, world','hello,world'),('1.5','15'),('01:02','01：02'),
                        ('1，5','1,5'),('12','1,2'),('https://a.com/x,y','https://a.com/x，y'),
                        ('【你好，】','【你好,】'),('你好}','你好）'),('<主体1>','<主体2>')]:
            self.assertEqual(candidate_operations(dict(base,snippet=old,replacement=new),old),[],(old,new))
        for text in ['me，}和me，}','【me，}】']:
            self.assertEqual(candidate_operations(dict(base,snippet='me，}',replacement='me,}'),text),[])
        for reason in ['建议将“me，}”改为“me,}”或“me。}”。','不建议将“me，}”改为“me,}”。',
                       '“me，}”改为“me,}”，另将“Now,”改为“Now，”。','前文示例“Now,”，检查“me，}”。']:
            self.assertEqual(candidate_operations(dict(base,snippet='me，}',reason=reason),'{Now, me，}'),[],reason)
        # A precise human-confirmed proposal is no longer blocked just because
        # the saved model classified punctuation as a grammar issue.
        self.assertTrue(candidate_operations(dict(base,kind='语法病句与语句不通顺',snippet='me，}',replacement='me,}'),'{me，}'))

    def test_multi_operation_dialogue_candidate_must_be_approved_and_applied_in_full(self):
        self.source.write_text(json.dumps({'id':'0001','caption':GOOD.replace('<主体1>向前跑。','<主体1>说道{Hello，world！}。')})+'\n')
        item=dict(semantic_issue('自然指令','Hello，world！','请人工确认标点。','标点符号硬伤'),can_auto_fix=False,replacement='Hello,world!')
        client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,IdentificationAgent(async_client=client),column='caption'))
        document=make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')
        issue=next(i for i in document['issues'] if i['kind']=='标点符号硬伤')
        self.assertEqual(len(issue['operations']),2)
        approved=decide(document,revision=0,issue_id=issue['issue_id'],decision='approve')
        self.assertIn('{Hello,world!}',next(iter(approved_changes(approved).values()))[0]['after'])
        duplicated=copy.deepcopy(approved)
        duplicate_issue=next(i for i in duplicated['issues'] if i['issue_id']==issue['issue_id'])
        duplicate_issue['operations'][1]=copy.deepcopy(duplicate_issue['operations'][0])
        with self.assertRaisesRegex(ValueError,'完整执行'):approved_changes(duplicated)
        next(i for i in approved['issues'] if i['issue_id']==issue['issue_id'])['operations'].pop()
        with self.assertRaisesRegex(ValueError,'完整执行'):approved_changes(approved)


if __name__=='__main__':unittest.main()
