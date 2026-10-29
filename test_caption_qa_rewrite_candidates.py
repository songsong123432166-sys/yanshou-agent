"""Concrete AI rewrites are proposals; human approval authorizes exact edits."""
import asyncio
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import caption_qa_agent as qa
from caption_qa_candidates import candidate_operations, enrich_candidates, REWRITE_POLICY
from caption_qa_review import SafeAutoFixAgent, make_review, approved_changes, decide, summary
from caption_qa_review_io import export_approved
from caption_qa_review_report import HEADERS, import_review
from test_caption_qa_upgrades import GOOD, FakeAsyncClient, semantic_reply, semantic_issue

OLD='镜头微微后拉顺时针缓慢从画面左侧<主体1>的头部开始的环绕'
NEW='镜头微微后拉、顺时针缓慢环绕，从画面左侧<主体1>的头部开始'
REASON='句式杂糅（“从……头部开始的环绕”与“从……头部开始环绕”混用），需人工重写为通顺表述，如“'+NEW+'”，涉及语序与结构调整，不作自动修复。'
KIND='语法病句与语句不通顺'


class RewriteCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input.jsonl';self.cache=self.root/'cache.jsonl'

    def build(self,structured=False,second=False):
        text='<主体1>在空中缓缓转为仰面朝上，'+OLD+'，并俯拍跟随<主体1>向下坠落。'
        if second:text+='镜头从向后拉，<主体1>展开双臂。'
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',text)
        self.source.write_text(json.dumps({'id':'cm4d97ef7d','caption':raw,'note':'其他字段保留'})+'\n')
        # Even a model's forged auto flag must not turn a rewrite automatic.
        item=dict(semantic_issue('自然指令',OLD,REASON,KIND),can_auto_fix=True)
        if structured:item['replacement']=NEW
        issues=[item]
        if second:issues.append(dict(semantic_issue('自然指令','镜头从向后拉','建议改为“镜头向后拉”。',KIND),replacement='镜头向后拉',can_auto_fix=False))
        client=FakeAsyncClient(semantic_reply(issues=issues,need_human_review=True))
        asyncio.run(qa.aprocess_batch(self.source,self.cache,SafeAutoFixAgent(async_client=client),column='caption'))
        return raw,make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def test_screenshot_and_structured_suggestion_are_pending_and_preview_the_full_rewrite(self):
        for structured in (False,True):
            self.cache.unlink(missing_ok=True)
            raw,doc=self.build(structured)
            self.assertEqual(json.loads(self.cache.read_text())['qa_revised_caption'],raw)
            self.assertEqual(approved_changes(doc),{})
            self.assertEqual(summary(doc)['needs_confirmation'],1)
            issue=doc['issues'][0]
            self.assertEqual(issue['decision'],'pending');self.assertFalse(issue['can_auto_fix'])
            self.assertTrue(issue['requires_confirmation']);self.assertEqual(issue['candidate_policy'],REWRITE_POLICY)
            self.assertEqual({(o['candidate_old'],o['candidate_new']) for o in issue['operations']},{(OLD,NEW)})

    def test_approval_applies_only_this_rewrite_and_keeps_source_id_other_fields_and_other_issue(self):
        raw,doc=self.build(second=True);first=doc['issues'][0]
        approved=decide(doc,revision=0,issue_id=first['issue_id'],decision='approve')
        source=self.source.read_bytes()
        result=export_approved(approved,self.source,self.source,self.cache,self.root,'grammar')
        self.assertEqual(json.loads(Path(result['corrected_path']).read_text()),
                         {'id':'cm4d97ef7d','caption':raw.replace(OLD,NEW),'note':'其他字段保留'})
        self.assertEqual(self.source.read_bytes(),source)
        self.assertEqual(summary(approved)['pending'],1)
        audit=json.loads(Path(result['recheck_path']).read_text())
        self.assertEqual(audit['changes'][0]['修改来源'],'approved_candidate')
        self.assertEqual(audit['model_requests'],0)
        self.assertEqual(approved['field_edits'],{})

    def test_no_concrete_rewrite_ambiguous_example_and_repeated_location_stay_manual(self):
        base={'kind':KIND,'field':'自然指令','snippet':OLD}
        for reason in ['句式杂糅，请人工调整语序。','应删改使句子通顺。',
                       f'可以改为“{NEW}”或者改为“镜头后拉”。',
                       f'例如“{NEW}”，也可例如“镜头后拉”。',
                       f'不建议改为“{NEW}”。']:
            self.assertEqual(candidate_operations(dict(base,reason=reason),OLD),[],reason)
        self.assertEqual(candidate_operations(dict(base,reason=REASON),OLD+'。'+OLD),[])

    def test_disabled_checks_literals_technical_tokens_and_numeric_changes_are_not_candidates(self):
        for old,new,kind in [('逐渐逐渐消散','逐渐消散',KIND),
                             ('树树','树','无歧义错别字'),
                             ('白袍','红袍','无歧义错别字'),
                             ('<主体1><主体1>站立','<主体1>站立',KIND),
                             ('<主体1>向前跑','<主体1>向后跑','属性冲突'),
                             ('【我来来】','【我来了】',KIND),
                             ('https://a.com/word','https://a.com/words',KIND),
                             ('第4秒，镜头后拉','第5秒，镜头后拉',KIND),
                             ('<主体1>站立','<主体2>站立',KIND)]:
            self.assertEqual(candidate_operations({'kind':kind,'field':'自然指令','snippet':old,'replacement':new,'reason':'请人工确认改稿。'},old),[],(old,new))

    def test_symbol_and_subject_link_suggestions_are_explicit_human_candidates(self):
        for old,new,kind in [('{HBO}','【HBO】','内容符号不规范'),
                             ('男子向前跑','<主体1>向前跑','隐式漏挂主体标签'),
                             ('主体1向前跑','<主体1>向前跑','裸标签漏写 <>'),
                             ('咳素','咳嗽','无歧义错别字')]:
            operations=candidate_operations({'kind':kind,'field':'自然指令','snippet':old,'replacement':new,'reason':'请人工确认此方案。'},old)
            self.assertTrue(operations,(old,new))
            revised=old
            for op in reversed(operations):revised=revised[:op['start']]+op['new']+revised[op['end']:]
            self.assertEqual(revised,new)
            self.assertTrue(all(o['review_only'] for o in operations))

    def test_candidate_requires_complete_unmodified_operations_and_human_approval(self):
        _,doc=self.build();issue=doc['issues'][0]
        approved=decide(doc,revision=0,issue_id=issue['issue_id'],decision='approve')
        for mutation in ('auto','partial','tamper'):
            corrupt=copy.deepcopy(approved);target=corrupt['issues'][0]
            if mutation=='auto':target['decision']='auto'
            elif mutation=='partial':target['operations'].pop()
            else:target['operations'][0]['new']='随意新增动作'
            with self.assertRaises(ValueError):approved_changes(corrupt)
        for decision in ('keep','reject','defer'):
            self.assertEqual(approved_changes(decide(doc,revision=0,issue_id=issue['issue_id'],decision=decision)),{})

    def test_old_pending_issues_enrich_once_without_changing_saved_decisions_or_manual_drafts(self):
        _,doc=self.build();issue=doc['issues'][0]
        issue['operations']=[];issue.pop('requires_confirmation');issue.pop('candidate_policy')
        for decision in ('approve','keep','reject','manual'):
            saved=copy.deepcopy(doc);saved['issues'][0]['decision']=decision
            self.assertEqual(enrich_candidates(saved),0)
        manual=copy.deepcopy(doc);manual['field_edits'][issue['field_id']]='最终稿'
        self.assertEqual(enrich_candidates(manual),0)
        self.assertEqual(enrich_candidates(doc),1);self.assertEqual(enrich_candidates(doc),0)
        self.assertEqual(issue['decision'],'pending');self.assertEqual(doc['events'],[])

    def test_review_csv_confirms_rewrite_without_manual_field_and_long_structured_candidate_is_valid(self):
        raw,doc=self.build();issue=doc['issues'][0];path=self.root/'review.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(HEADERS)
            writer.writerow([issue['issue_id'],issue['sample_id'],issue['field'],issue['label'],'\n'.join(issue['excerpts']),issue['subject'],'确认按建议改',''])
        imported=import_review(doc,path,revision=0)
        self.assertEqual(imported['field_edits'],{})
        self.assertIn(NEW,next(iter(approved_changes(imported).values()))[0]['after'])
        sections=qa.parse_caption_sections(raw)
        item=dict(semantic_issue('自然指令',OLD,REASON,KIND),can_auto_fix=False,replacement=NEW+'，镜头后拉'*30)
        qa._validate_semantic(semantic_reply(issues=[item]),sections)
        item['replacement']='字'*4001
        with self.assertRaisesRegex(ValueError,'4000'):qa._validate_semantic(semantic_reply(issues=[item]),sections)


if __name__=='__main__':unittest.main()
