"""Human-confirmed advice shares one executable path across enabled checks."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import captionqa.caption_qa_agent as qa
import captionqa.caption_qa_web as web
from captionqa.caption_qa_candidates import candidate_operations
from captionqa.caption_qa_review import IdentificationAgent, make_review, decide, approved_changes
from captionqa.caption_qa_suggestions import operations, prepare_proposal, POLICY
from tests.test_caption_qa_upgrades import GOOD, FakeSyncClient, FakeAsyncClient, semantic_reply, semantic_issue


def apply(text,ops):
    for op in reversed(ops):text=text[:op['start']]+op['new']+text[op['end']:]
    return text


class SuggestionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input.jsonl';self.cache=self.root/'cache.jsonl'

    def test_saved_suggestion_forms_resolve_to_exact_expected_text_across_issue_types(self):
        cases=[('镜头切换(背景音乐结束)','镜头切换(背景音乐结束)',
                '背景音乐特征应使用中文全角括号，将“镜头切换(背景音乐结束)”改为“镜头切换（背景音乐结束）”。','镜头切换（背景音乐结束）','标点符号硬伤'),
               ('<图片1>:','<图片1>:','<图片1>：','<图片1>：','标点符号硬伤'),
               ('棘刺,有獠牙','棘刺,有獠牙','棘刺，有獠牙','棘刺，有獠牙','标点符号硬伤'),
               ('{你好。}','你好。}','段尾闭合符后补句号，改为「你好。}。」。','{你好。}。','标点符号硬伤'),
               ('<主体2:>站立。','<主体2:>','冒号移出，改为"<主体2>: "，其余文字不变。','<主体2>: 站立。','标点符号硬伤'),
               ('镜头切换，，中景','镜头切换，，中景','删除重复逗号，改为“镜头切换，中景”。','镜头切换，中景','标点符号硬伤'),
               ('<图片4>:的近景','<图片4>:的近景','删除多余的冒号，改为“<图片4>的近景”。','<图片4>的近景','标点符号硬伤'),
               ('<发出一声响>第12秒','<发出一声响>第12秒','在音效闭合符 > 与“第12秒”之间补一个句号。','<发出一声响>。第12秒','标点符号硬伤'),
               ('<主体1>与<主体2的>的剑刃交击','<主体2的>','将“的”移出，改为“<主体1>与<主体2>的剑刃交击”。','<主体1>与<主体2>的剑刃交击','内容符号不规范'),
               ('全景拍摄<主体2>张发出嘶吼声','全景拍摄<主体2>张发出嘶吼声',
                '“张发出嘶吼声”成分残缺（疑为“张嘴发出嘶吼声”等），需人工确认。','全景拍摄<主体2>张嘴发出嘶吼声','语法病句与语句不通顺'),
               ('冷峻地问到','冷峻地问到','“问到”位于引语前，疑为“问道”；转人工确认。','冷峻地问道','无歧义错别字'),
               ('握着黑色话举至唇前','握着黑色话举至唇前','“黑色话”疑为“黑色话筒”漏字，转人工。','握着黑色话筒举至唇前','无歧义错别字'),
               ('背景音乐响起氛一段围音乐','背景音乐响起氛一段围音乐','疑为“背景音乐响起一段氛围音乐”，需人工确认。','背景音乐响起一段氛围音乐','语法病句与语句不通顺'),
               ('从<主体4>第右前侧','从<主体4>第右前侧','“第右前侧”疑似“的右前侧”的误写。','从<主体4>的右前侧','无歧义错别字'),
               ('写有“甲地”和“乙名”的牌子','写有“甲地”和“乙名”的牌子','画面文字应使用【甲地】【乙名】，符号变更需确认。','写有【甲地】和【乙名】的牌子','内容符号不规范'),
               ('伴有清晰的呼吸声','伴有清晰的呼吸声','按规范应以尖括号包裹（如 <伴有清晰的呼吸声>）。','<伴有清晰的呼吸声>','内容符号不规范')]
        for text,snippet,reason,expected,kind in cases:
            issue={'kind':kind,'field':'自然指令','snippet':snippet,'reason':reason}
            ops=candidate_operations(issue,text)
            self.assertTrue(ops,(text,reason));self.assertEqual(apply(text,ops),expected)
            self.assertTrue(all(o['review_only'] for o in ops))

    def test_repeated_snippet_is_changed_only_when_advice_explicitly_says_all_places(self):
        issue={'kind':'标点符号硬伤','field':'自然指令','snippet':'(背景音乐继续)',
               'reason':'音乐应使用全角括号，将(背景音乐继续)改为（背景音乐继续）。共出现3处，均同此修改。'}
        text='(背景音乐继续)，镜头后拉。(背景音乐继续)，镜头前推。(背景音乐继续)。'
        ops=candidate_operations(issue,text)
        self.assertEqual(len(ops),6)
        self.assertEqual(apply(text,ops),text.replace('(','（').replace(')','）'))
        issue['reason']='改为“（背景音乐继续）”。'
        self.assertEqual(candidate_operations(issue,text),[])

    def test_ambiguous_partial_examples_disabled_checks_and_unrelated_quotes_are_not_executed(self):
        base={'kind':'语法病句与语句不通顺','field':'自然指令','snippet':'将身体向前'}
        for reason in ['缺少动作，如“倾”“探”，需人工补全。','疑为“向前/向后倾”，需确认。',
                       '“将身体向前”改为“将身体向前倾”或“将身体向前探”。',
                       '请重写通顺。','不建议改为“向前倾”。']:
            self.assertEqual(candidate_operations(dict(base,reason=reason),'将身体向前'),[],reason)
        for kind in ['属性冲突','重复标签前缀','衍字/叠词冗余']:
            self.assertEqual(candidate_operations(dict(base,kind=kind,replacement='向前倾',reason='明确改为“向前倾”。'),'将身体向前'),[])

    def build(self):
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。','<主体1>向前跑，镜头从向后拉。')
        self.source.write_text(json.dumps({'id':'0007','caption':raw,'note':'不改其他列'})+'\n')
        item=dict(semantic_issue('自然指令','镜头从向后拉','句式不通顺，请按原意改写。','语法病句与语句不通顺'),can_auto_fix=False)
        client=FakeAsyncClient(semantic_reply(issues=[item],need_human_review=True))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,IdentificationAgent(async_client=client),column='caption'))
        doc=make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')
        app=web.App(self.root/'out',self.root/'jobs');self.addCleanup(app.close)
        job=web.Job('a'*32,'input.jsonl',self.source,self.root/'out/ids.txt',self.cache,False,'caption',None,8,status='done')
        job.selected_column='caption';job.id_column='id';app.jobs[job.id]=job;app._save_review(job,doc)
        return app,job,doc,raw

    def test_generated_draft_is_saved_pending_and_only_confirmation_applies_it(self):
        app,job,doc,raw=self.build();issue=doc['issues'][0]
        client=FakeSyncClient({'ready':True,'source':'镜头从向后拉','replacement':'镜头向后拉','explanation':'删除错误介词。'})
        original_prepare=prepare_proposal
        with patch.object(web,'read_saved_key',return_value='offline-test'),patch('captionqa.caption_qa_suggestions.prepare_proposal',
            side_effect=lambda i,f,**kw:original_prepare(i,f,client=client,**kw)):
            result=app.propose_review(job.id,doc['revision'],issue['issue_id'])
        self.assertEqual(len(client.calls),1)
        updated=result['review'];candidate=updated['issues'][0]
        self.assertEqual(candidate['decision'],'pending');self.assertTrue(candidate['requires_confirmation'])
        self.assertEqual(approved_changes(updated),{});self.assertEqual(updated['field_edits'],{})
        approved=decide(updated,revision=updated['revision'],issue_id=issue['issue_id'],decision='approve')
        self.assertIn('镜头向后拉',next(iter(approved_changes(approved).values()))[0]['after'])
        self.assertEqual(json.loads(self.source.read_text())['caption'],raw)
        self.assertEqual(json.loads(self.cache.read_text())['qa_revised_caption'],raw)

    def test_generation_failure_stale_revision_and_saved_decision_never_overwrite_review(self):
        app,job,doc,_=self.build();issue=doc['issues'][0];path=job.cache.with_suffix('.review.json');before=path.read_bytes()
        with patch.object(web,'read_saved_key',return_value='offline-test'),patch('captionqa.caption_qa_suggestions.prepare_proposal',side_effect=ValueError('缺少依据')):
            with self.assertRaisesRegex(ValueError,'缺少依据'):app.propose_review(job.id,doc['revision'],issue['issue_id'])
        self.assertEqual(path.read_bytes(),before)
        with patch('captionqa.caption_qa_suggestions.prepare_proposal') as mock:
            with self.assertRaisesRegex(ValueError,'版本'):app.propose_review(job.id,99,issue['issue_id'])
            mock.assert_not_called()
        kept=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],decision='keep');app._save_review(job,kept)
        with patch('captionqa.caption_qa_suggestions.prepare_proposal') as mock:
            with self.assertRaisesRegex(ValueError,'已有审核结论'):app.propose_review(job.id,kept['revision'],issue['issue_id'])
            mock.assert_not_called()

    def test_model_refusal_malformed_output_and_unrelated_edit_are_rejected(self):
        issue={'kind':'语法病句与语句不通顺','reason':'按原意修改。','snippet':'镜头从向后拉'}
        field={'name':'自然指令','original':'<主体1>向前跑，镜头从向后拉。'}
        values=[{'ready':False,'explanation':'存在多种原意。'},
                {'ready':True,'source':'不存在','replacement':'改稿','explanation':''},
                {'ready':True,'source':field['original'],'replacement':'<主体1>向后跑，镜头向后拉。','explanation':''},
                {'ready':'true','source':'镜头从向后拉','replacement':'镜头向后拉','explanation':''}]
        for value in values:
            with self.assertRaises(ValueError):prepare_proposal(issue,field,api_key='test',base_url='offline',model='mock',client=FakeSyncClient(value))
        from tests.test_caption_qa_upgrades import sdk_response
        client=FakeSyncClient();response=sdk_response({'ready':True,'source':'镜头从向后拉','replacement':'镜头向后拉','explanation':''})
        response.choices[0].finish_reason='length'
        with patch.object(client,'create',return_value=response):
            with self.assertRaisesRegex(ValueError,'生成改稿失败'):prepare_proposal(issue,field,api_key='test',base_url='offline',model='mock',client=client)

    def test_concurrent_review_change_discards_generated_candidate(self):
        app,job,doc,_=self.build();issue=doc['issues'][0]
        def prepare(*args,**kw):
            kept=decide(doc,revision=doc['revision'],issue_id=issue['issue_id'],decision='keep');app._save_review(job,kept)
            return {'source':'镜头从向后拉','replacement':'镜头向后拉','start':next(f for f in doc['fields'] if f['field_id']==issue['field_id'])['original'].index('镜头从向后拉')}
        with patch.object(web,'read_saved_key',return_value='test'),patch('captionqa.caption_qa_suggestions.prepare_proposal',side_effect=prepare):
            with self.assertRaisesRegex(ValueError,'生成期间审核版本'):app.propose_review(job.id,doc['revision'],issue['issue_id'])
        self.assertEqual(json.loads(job.cache.with_suffix('.review.json').read_text())['issues'][0]['decision'],'keep')

    def test_exact_audio_word_candidate_is_human_only_and_marked_for_audio_review(self):
        issue={'kind':'无歧义错别字','field':'自然指令','snippet':'失帧电吉他','reason':'疑为“失真电吉他”，保留原文转人工确认。'}
        text='（背景音乐是失帧电吉他演奏）。'
        ops=candidate_operations(issue,text)
        self.assertTrue(ops);self.assertTrue(all(o['review_only'] for o in ops))
        self.assertEqual(ops[0]['candidate_scope'],'音频正文')
        self.assertEqual(apply(text,ops),'（背景音乐是失真电吉他演奏）。')


if __name__=='__main__':unittest.main()
