"""Exercise hung requests, retries and resuming with a fake API only."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_review import SafeAutoFixAgent
from test_caption_qa_upgrades import GOOD, semantic_reply, sdk_response


class SlowClient:
    def __init__(self, *, all_slow=False, first_error=False):
        self.chat=SimpleNamespace(completions=self)
        self.calls=[]
        self.cancelled=0
        self.all_slow=all_slow
        self.first_error=first_error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.first_error and len(self.calls)==1:
            raise TimeoutError('do not echo this request')
        sections=json.loads(kwargs['messages'][1]['content'])['caption_sections']
        if self.all_slow or '慢' in sections.get('提示',''):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled+=1
                raise
        return sdk_response(semantic_reply())


class RequestProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_total_deadline_cancels_api_and_keeps_local_fixes(self):
        client=SlowClient(all_slow=True)
        agent=SafeAutoFixAgent(async_client=client, timeout=1, total_timeout=.025)
        report=await agent.acheck_single_caption(GOOD.replace('身穿白袍。','身穿白袍'))
        self.assertFalse(report['semantic_qa_performed'])
        self.assertIn('总时限',report['human_review_reason'])
        self.assertNotEqual(report['qa_scope'],'input_error')
        self.assertIn('身穿白袍。',report['revised_caption'])
        self.assertTrue(report['need_human_review'])
        self.assertEqual(client.cancelled,1)

    async def test_request_timeout_retries_and_reports_safe_state(self):
        client=SlowClient(all_slow=True)
        events=[]
        token=qa._REQUEST_PROGRESS.set(events.append)
        try:
            report=await qa.arun_llm_semantic_qa(qa.parse_caption_sections(GOOD),[],client=client,
                timeout=.01,total_timeout=.2,max_retries=1,retry_delay=0)
        finally:
            qa._REQUEST_PROGRESS.reset(token)
        self.assertFalse(report['performed'])
        self.assertEqual(len(client.calls),2)
        self.assertEqual(client.cancelled,2)
        self.assertEqual([e['attempt'] for e in events if e['state']=='requesting'],[1,2])
        self.assertEqual([e['state'] for e in events],['requesting','retry_wait','requesting'])
        self.assertNotIn('do not echo',json.dumps(events))

    async def test_backoff_counts_toward_total_deadline(self):
        client=SlowClient(first_error=True)
        report=await qa.arun_llm_semantic_qa(qa.parse_caption_sections(GOOD),[],client=client,
            timeout=1,total_timeout=.015,max_retries=2,retry_delay=1)
        self.assertFalse(report['performed'])
        self.assertIn('总时限',report['human_review_reason'])
        self.assertEqual(len(client.calls),1)

    async def test_completion_order_progress_ids_and_only_timeout_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'in.jsonl';cache=root/'result.jsonl'
            records=[{'id':name,'caption':GOOD+'## 提示\n'+name+'。\n'} for name in ('慢','快')]
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            original=source.read_bytes();events=[]
            agent=SafeAutoFixAgent(async_client=SlowClient(),timeout=1,total_timeout=.04)
            await qa.aprocess_batch(source,cache,agent,concurrency=2,on_progress=events.append)
            rows=[json.loads(line) for line in cache.read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows],['快','慢'])
            self.assertTrue(rows[0]['qa_checkpoint']['completed'])
            self.assertFalse(rows[1]['qa_checkpoint']['completed'])
            self.assertTrue(rows[1]['qa_checkpoint']['request_stats']['timed_out'])
            self.assertEqual(rows[1]['qa_checkpoint']['request_stats']['attempts'],1)
            self.assertEqual({e['sample_id'] for e in events},{'快','慢'})
            for name in ('快','慢'):
                own=[e for e in events if e['sample_id']==name]
                self.assertEqual(own[0]['state'],'checking')
                self.assertEqual(own[-1]['state'],'finished')
                self.assertEqual({e['row_index'] for e in own},{1 if name=='慢' else 2})
            self.assertIsNone(qa._REQUEST_PROGRESS.get())
            from test_caption_qa_upgrades import FakeAsyncClient
            retry_client=FakeAsyncClient(semantic_reply())
            retried=SafeAutoFixAgent(async_client=retry_client)
            self.assertEqual(await qa.aprocess_batch(source,cache,retried),1)
            self.assertEqual(len(retry_client.calls),1)
            self.assertEqual(source.read_bytes(),original)
            job=web.Job('a'*32,'in.jsonl',source,root/'out.txt',cache,False,None,None,2)
            self.assertEqual(job.snapshot()['timed_out_count'],0)

    async def test_cancellation_propagates_without_completing_checkpoint(self):
        client=SlowClient(all_slow=True)
        agent=qa.CaptionQAAgent(async_client=client,total_timeout=1)
        task=asyncio.create_task(agent.acheck_single_caption(GOOD))
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(client.cancelled,1)

    async def test_telemetry_callback_failure_does_not_fail_check(self):
        from test_caption_qa_upgrades import FakeAsyncClient
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'in.jsonl';cache=Path(folder)/'out.jsonl'
            source.write_text(json.dumps({'id':'001','caption':GOOD})+'\n')
            def broken(event):
                raise ValueError('UI disconnected')
            agent=qa.CaptionQAAgent(async_client=FakeAsyncClient(semantic_reply()))
            await qa.aprocess_batch(source,cache,agent,on_progress=broken)
            self.assertTrue(json.loads(cache.read_text())['qa_checkpoint']['completed'])


class WebRequestProgressTests(unittest.TestCase):
    def test_live_state_and_timeout_survive_reload_and_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);app=web.App(root/'out',root/'work')
            self.addCleanup(app.close)
            import csv
            stream=io.StringIO();writer=csv.writer(stream)
            writer.writerow(['id','caption']);writer.writerow(['0001',GOOD])
            raw=stream.getvalue().encode()
            client=SlowClient(all_slow=True)
            from test_caption_qa_upgrades import FakeAsyncClient
            def factory(**options):
                self.assertEqual(options['timeout'],60)
                self.assertEqual(options['total_timeout'],180)
                self.assertEqual(options['max_retries'],2)
                return SafeAutoFixAgent(async_client=client,timeout=1,total_timeout=.03)
            observed=[]
            real_batch=qa.aprocess_batch
            async def track(*args,**kwargs):
                callback=kwargs.get('on_progress')
                if callback:
                    def wrapped(event):
                        callback(event)
                        observed.append(app.jobs[item_id[0]].snapshot()['active_requests'])
                    kwargs['on_progress']=wrapped
                return await real_batch(*args,**kwargs)
            # Hold the executor until the created job id is available to the observer.
            import threading
            gate=threading.Event();app.executor.submit(gate.wait);item_id=[]
            with patch.object(web,'read_saved_key',return_value='fake'),patch(
                'caption_qa_review.SafeAutoFixAgent',side_effect=factory),patch.object(qa,'aprocess_batch',side_effect=track):
                item=app.create_job('test.csv',io.BytesIO(raw),len(raw),dry_run=False)
                item_id.append(item['id']);gate.set()
                app.executor.submit(lambda:None).result(timeout=20)
            job=app.jobs[item['id']]
            self.assertEqual(job.status,'partial')
            self.assertEqual(job.snapshot()['timed_out_count'],1)
            self.assertEqual(job.snapshot()['pending_retry'],1)
            self.assertEqual(job.snapshot()['active_requests'],[])
            self.assertTrue(any(rows and rows[0]['sample_id']=='0001' and rows[0]['attempt']==1
                                for rows in observed if rows and 'attempt' in rows[0]))
            app.save_history()
            restored=web.App(root/'out',root/'work');self.addCleanup(restored.close)
            self.assertEqual(restored.jobs[job.id].snapshot()['timed_out_count'],1)
            with patch.object(web,'read_saved_key',return_value='fake'),patch(
                'caption_qa_review.SafeAutoFixAgent',side_effect=lambda **kw: SafeAutoFixAgent(
                    async_client=FakeAsyncClient(semantic_reply()))):
                app.retry(job.id)
                app.executor.submit(lambda:None).result(timeout=20)
            self.assertEqual(job.snapshot()['timed_out_count'],0)
            self.assertEqual(job.status,'done')


if __name__=='__main__':unittest.main()
