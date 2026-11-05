import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import openpyxl
from captionqa.caption_column_selector import caption_candidates,choose_caption_with_ai,SAMPLE_CHARS
from captionqa.caption_field_jobs import FieldJobs
from tests.test_caption_field_normalization import STANDARD

class Client:
    def __init__(self,result=None,mutate=None,error=None,finish='stop'):
        self.chat=self;self.completions=self;self.calls=[];self.closes=0
        self.result=result if result is not None else {'decision':'selected','candidate_id':'c0'}
        self.mutate=mutate;self.error=error;self.finish=finish
    async def create(self,**options):
        self.calls.append(options)
        if self.error:raise self.error
        if self.mutate:self.mutate()
        content=self.result if isinstance(self.result,str) else json.dumps(self.result)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish,message=SimpleNamespace(content=content,refusal=None))])
    async def close(self):self.closes+=1

class SelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def source(self,records=None):
        path=self.root/'input.jsonl'
        rows=records if records is not None else [{'题目ID':'do-not-send-123','自定义正文':STANDARD,'机标caption':STANDARD,'备注':STANDARD,'是否运镜':'是'}]
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows));return path
    async def test_ai_selects_unusual_header_using_only_caption_candidates(self):
        source=self.source();client=Client()
        self.assertEqual(await choose_caption_with_ai(source,client=client,model='test'),(None,'自定义正文'))
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        self.assertEqual(len(payload['candidates']),1)
        value=payload['candidates'][0]
        self.assertEqual(value['header'],'自定义正文');self.assertIn('主体',value['fields'])
        encoded=json.dumps(payload,ensure_ascii=False)
        for text in ['do-not-send-123','机标caption','备注','是否运镜']:self.assertNotIn(text,encoded)
    async def test_sample_count_and_size_are_bounded(self):
        text=STANDARD+('长内容'*1000)
        candidates=caption_candidates(self.source([{'正文':text+str(i)} for i in range(20)]))
        self.assertEqual(len(candidates[0]['samples']),3)
        self.assertTrue(all(len(s)<=SAMPLE_CHARS for s in candidates[0]['samples']))
    async def test_machine_group_children_are_excluded(self):
        source=self.root/'grouped.xlsx';book=openpyxl.Workbook();ws=book.active
        ws.append(['ID','机器标注参考',None,'人工标注',None]);ws.merge_cells('B1:C1');ws.merge_cells('D1:E1')
        ws.append(['','运镜','caption','是否运镜','caption']);ws.append(['id','后移',STANDARD,'是',STANDARD]);book.save(source);book.close()
        candidates=caption_candidates(source)
        self.assertEqual([c['column'] for c in candidates],['__column_5'])
    async def test_unknown_model_column_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'不在候选范围'):
            await choose_caption_with_ai(self.source(),client=Client({'decision':'selected','candidate_id':'机标caption'}),model='test')
    async def test_internal_review_group_is_excluded(self):
        candidates=caption_candidates(self.source([{'正文':STANDARD,'内部抽检':STANDARD}]))
        self.assertEqual([c['column'] for c in candidates],['正文'])
    async def test_ai_cannot_override_explicit_human_group_with_raw_prompt(self):
        source=self.source([{'原始文本':STANDARD,'人工标注':STANDARD}])
        with self.assertRaisesRegex(ValueError,'人工标注表头不一致'):
            await choose_caption_with_ai(source,client=Client(),model='test')
    async def test_uncertain_result_requests_manual_selection(self):
        with self.assertRaisesRegex(ValueError,'无法唯一确定'):
            await choose_caption_with_ai(self.source(),client=Client({'decision':'uncertain','candidate_id':None}),model='test')
    async def test_malformed_and_truncated_responses_are_rejected(self):
        for client in [Client('not json'),Client([]),Client(finish='length')]:
            with self.subTest(client=client),self.assertRaisesRegex(ValueError,'未成功'):
                await choose_caption_with_ai(self.source(),client=client,model='test')
    async def test_transport_error_is_sanitized(self):
        with self.assertRaisesRegex(ValueError,'未成功') as caught:
            await choose_caption_with_ai(self.source(),client=Client(error=TimeoutError('private token detail')),model='test')
        self.assertNotIn('private',str(caught.exception))
    async def test_non_caption_data_does_not_trigger_ai(self):
        client=Client()
        with self.assertRaisesRegex(ValueError,'未调用 AI'):
            await choose_caption_with_ai(self.source([{'ID':'001','是否运镜':'是','备注':STANDARD}]),client=client,model='test')
        self.assertEqual(client.calls,[])
    async def test_source_change_invalidates_model_result(self):
        source=self.source();client=Client(mutate=lambda:source.write_text(source.read_text()+'\n'))
        with self.assertRaisesRegex(ValueError,'文件发生变化'):
            await choose_caption_with_ai(source,client=client,model='test')
    async def test_normal_form_column_limit_is_enforced_before_sampling(self):
        with self.assertRaisesRegex(ValueError,'列数较多'):
            caption_candidates(self.source([{str(i):'text' for i in range(257)}]))
    async def test_too_many_candidates_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'较多 Caption 候选'):
            caption_candidates(self.source([{f'内容{i}':STANDARD for i in range(17)}]))

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.client=Client();self.manager=FieldJobs(self.root/'jobs',self.root/'out',lambda:self.client);self.addCleanup(self.manager.close)
    def create(self,row,**options):
        data=(json.dumps(row,ensure_ascii=False)+'\n').encode()
        result=self.manager.create('test.jsonl',io.BytesIO(data),len(data),**options)
        self.manager.executor.submit(lambda:None).result(timeout=10)
        return self.manager.jobs[result['id']]
    def test_ambiguous_name_uses_ai_and_preserves_source(self):
        row={'题目ID':'001','自定义正文':STANDARD,'备注':'unchanged'}
        job=self.create(row)
        self.assertEqual(job['status'],'done',job['error']);self.assertEqual(job['selected_column'],'自定义正文')
        self.assertEqual(job['column_detection'],'ai');self.assertEqual(job['selection_model_requests'],1)
        self.assertEqual(job['model_requests'],1);self.assertEqual(len(self.client.calls),1)
        self.assertEqual(json.loads(Path(job['outputs']['corrected']).read_text()),row)
        self.assertEqual(self.client.closes,2)
    def test_rule_identification_never_calls_ai(self):
        job=self.create({'id':'001','caption':STANDARD})
        self.assertEqual(job['status'],'done');self.assertEqual(job['column_detection'],'rules');self.assertEqual(self.client.calls,[])
    def test_manual_column_is_not_overridden_or_sent_to_ai(self):
        job=self.create({'id':'001','自定义正文':STANDARD},column='自定义正文')
        self.assertEqual(job['status'],'done');self.assertEqual(self.client.calls,[])
    def test_invalid_manual_column_fails_without_ai(self):
        job=self.create({'id':'001','自定义正文':STANDARD},column='不存在')
        self.assertEqual(job['status'],'failed');self.assertEqual(self.client.calls,[])
    def test_uncertain_ai_does_not_generate_outputs(self):
        self.client.result={'decision':'uncertain','candidate_id':None}
        job=self.create({'id':'001','正文A':STANDARD,'正文B':STANDARD})
        self.assertEqual(job['status'],'failed');self.assertEqual(job['outputs'],{})
        self.assertEqual(job['selection_model_requests'],1);self.assertEqual(self.client.closes,1)

if __name__=='__main__':unittest.main()
