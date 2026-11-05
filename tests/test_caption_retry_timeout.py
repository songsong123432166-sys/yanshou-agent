import asyncio,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import captionqa.caption_qa_web as web
from tests import test_caption_qa_web as fixtures
from tests.test_caption_qa_web import csv_bytes
from tests.test_caption_qa_upgrades import GOOD

class RetryTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.app=web.App(self.root/'out',self.root/'work');self.addCleanup(self.app.close)
        data=csv_bytes([('001',GOOD),('002',GOOD)])
        result=self.app.create_job('timeout.csv',io.BytesIO(data),len(data),dry_run=True,review_first=False)
        self.finish();self.job=self.app.jobs[result['id']]
    def finish(self):self.app.executor.submit(lambda:None).result(timeout=30)
    def test_longer_retry_wait_preserves_completed_cache_and_total_budget(self):
        before=self.job.cache.read_bytes();self.job.status='partial'
        real=web.qa.CaptionQAAgent;options=[]
        def factory(**kw):options.append(kw);return real(**kw)
        with patch.object(web.qa,'CaptionQAAgent',side_effect=factory):
            self.app.retry(self.job.id,concurrency=2,request_timeout_seconds=120);self.finish()
        self.assertEqual(self.job.status,'done',self.job.error)
        self.assertEqual(options[0]['timeout'],120);self.assertIsNone(options[0]['total_timeout'])
        self.assertEqual(self.job.snapshot()['request_timeout_seconds'],120)
        self.assertEqual(self.job.concurrency,2);self.assertEqual(self.job.cache.read_bytes(),before)
    def test_invalid_timeout_is_rejected_without_mutating_job(self):
        for value in [0,901,True,1.5,'120']:
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'等待须为'):
                self.app.retry(self.job.id,request_timeout_seconds=value)
        self.assertEqual(self.job.request_timeout_seconds,60)
    def test_http_retry_accepts_longer_timeout_and_preserves_cache(self):
        before=self.job.cache.read_bytes();self.job.status='partial'
        status,body=fixtures.WebTests.request(self,'POST',f'/api/jobs/{self.job.id}/retry',body=b'{"concurrency":2,"request_timeout_seconds":120}')
        self.assertEqual(status,202,body);self.finish()
        self.assertEqual(self.job.request_timeout_seconds,120);self.assertEqual(self.job.cache.read_bytes(),before)
    def test_timeout_survives_history_restore(self):
        self.job.status='partial';self.app.retry(self.job.id,request_timeout_seconds=120);self.finish()
        restored=web.App(self.root/'out',self.root/'work');self.addCleanup(restored.close)
        self.assertEqual(restored.jobs[self.job.id].request_timeout_seconds,120)
    def test_extended_budget_keeps_rules_and_completed_records(self):
        before=self.job.cache.read_bytes();self.job.status='partial'
        real=web.qa.CaptionQAAgent;options=[]
        def factory(**kw):options.append(kw);return real(**kw)
        with patch.object(web.qa,'CaptionQAAgent',side_effect=factory):
            status,body=fixtures.WebTests.request(self,'POST',f'/api/jobs/{self.job.id}/retry',body=b'{"request_timeout_seconds":600,"caption_timeout_seconds":null,"concurrency":2}')
            self.assertEqual(status,202,body);self.finish()
        self.assertEqual(options[0]['timeout'],600);self.assertIsNone(options[0]['total_timeout'])
        self.assertEqual(self.job.cache.read_bytes(),before)
        restored=web.App(self.root/'out',self.root/'work');self.addCleanup(restored.close)
        self.assertIsNone(restored.jobs[self.job.id].caption_timeout_seconds)
        self.assertEqual(restored.jobs[self.job.id].request_timeout_seconds,600)
    def test_total_deadline_cannot_be_reenabled(self):
        for value in [0,180,600,True,1.5,'600']:
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'总时限已取消'):
                self.app.retry(self.job.id,caption_timeout_seconds=value)
        self.assertEqual(self.job.concurrency,8);self.assertEqual(self.job.request_timeout_seconds,60)
        self.assertIsNone(self.job.caption_timeout_seconds)
    def test_legacy_total_deadline_is_discarded_on_restore(self):
        self.app.save_history();p=self.root/'work/session.json';data=json.loads(p.read_text())
        data['jobs'][0].update(caption_timeout_seconds=180,request_timeout_seconds=600);p.write_text(json.dumps(data))
        restored=web.App(self.root/'out',self.root/'work');self.addCleanup(restored.close)
        job=restored.jobs[self.job.id]
        self.assertIsNone(job.caption_timeout_seconds);self.assertEqual(job.request_timeout_seconds,600)
        self.assertIsNone(job.snapshot()['caption_timeout_seconds'])
    def test_older_history_uses_default_timeout(self):
        self.app.save_history();p=self.root/'work/session.json';data=json.loads(p.read_text())
        data['jobs'][0].pop('request_timeout_seconds',None);p.write_text(json.dumps(data))
        restored=web.App(self.root/'out',self.root/'work');self.addCleanup(restored.close)
        self.assertEqual(restored.jobs[self.job.id].request_timeout_seconds,60)

class NoAggregateDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_agent_uses_only_per_request_timeout(self):
        from tests.test_caption_qa_upgrades import FakeAsyncClient,semantic_reply
        agent=web.qa.CaptionQAAgent(async_client=FakeAsyncClient(semantic_reply()),timeout=.2)
        self.assertIsNone(agent.total_timeout)
        real=asyncio.wait_for
        with patch.object(web.qa.asyncio,'wait_for',wraps=real) as waits:
            report=await agent.acheck_single_caption(GOOD)
        self.assertTrue(report['semantic_qa_performed'])
        self.assertEqual([call.kwargs['timeout'] for call in waits.call_args_list],[.2])
    async def test_hung_request_still_times_out_and_retries_finitely(self):
        from tests.test_caption_qa_request_progress import SlowClient
        client=SlowClient(all_slow=True)
        agent=web.qa.CaptionQAAgent(async_client=client,timeout=.01,max_retries=1,retry_delay=0)
        report=await agent.acheck_single_caption(GOOD)
        self.assertFalse(report['semantic_qa_performed']);self.assertEqual(len(client.calls),2)
        self.assertEqual(client.cancelled,2)
        self.assertNotIn('总时限',report['human_review_reason'])

if __name__=='__main__':unittest.main()
