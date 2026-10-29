"""Old alias false positives are retired without deleting source or human history."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import caption_qa_web as web
from test_caption_qa_upgrades import GOOD


class AliasHistoryTests(unittest.TestCase):
    def test_english_alias_v2_cache_cannot_resume_or_export_stale_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);app=web.App(root/'out',root/'work');self.addCleanup(app.close)
            raw=GOOD.replace('## 自然指令','## natural_instruction')
            payload=(json.dumps({'id':'0001','caption':raw},ensure_ascii=False)+'\n').encode()
            with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
                created=app.create_job('english.jsonl',io.BytesIO(payload),len(payload),dry_run=True)
                app.executor.submit(lambda:None).result(timeout=10)
            job=app.jobs[created['id']];self.assertEqual(job.status,'done',job.error)
            row=json.loads(job.cache.read_text());row['qa_report']['caption_parser_version']=2
            job.cache.write_text(json.dumps(row,ensure_ascii=False)+'\n');job.offset=0;job.rows.clear()
            old_cache=job.cache;old_bytes=old_cache.read_bytes();old_output=job.output
            old_review=job.cache.with_suffix('.review.json');review_bytes=old_review.read_bytes()
            self.assertTrue(job.snapshot()['outdated_instruction_parser'])
            with self.assertRaisesRegex(ValueError,'旧版未识别'):app.review(job.id)
            job.outdated_instruction_rows.clear();job.offset=0
            with self.assertRaises(ValueError):app.export_repair_report(job.id)
            context={k:v for k,v in row['qa_checkpoint'].items() if k not in {'key','completed'}}
            with self.assertRaisesRegex(ValueError,'旧版字段归属解析'):
                web.qa._load_completed_checkpoints(job.cache,context)
            with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
                app.retry(job.id);app.executor.submit(lambda:None).result(timeout=10)
            self.assertEqual(job.status,'done',job.error)
            self.assertNotEqual(job.cache,old_cache);self.assertNotEqual(job.output,old_output)
            self.assertEqual(old_cache.read_bytes(),old_bytes)
            self.assertEqual(old_review.read_bytes(),review_bytes)
            self.assertFalse(job.snapshot()['outdated_instruction_parser'])
            self.assertEqual(job.id_summary['instruction_subject_alerts'],[])

    def test_alias_cache_is_flagged_and_retry_is_local_with_fresh_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);app=web.App(root/'out',root/'work');self.addCleanup(app.close)
            raw=GOOD.replace('## 自然指令','## 自然描述')
            payload=(json.dumps({'id':'0001','caption':raw},ensure_ascii=False)+'\n').encode()
            with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
                created=app.create_job('alias.jsonl',io.BytesIO(payload),len(payload),dry_run=True)
                app.executor.submit(lambda:None).result(timeout=10)
            job=app.jobs[created['id']];self.assertEqual(job.status,'done',job.error)
            row=json.loads(job.cache.read_text());row['qa_report'].pop('caption_parser_version',None)
            old=row['qa_report'];old['subject_check']['missing_in_instruction']=['<主体1>']
            job.cache.write_text(json.dumps(row,ensure_ascii=False)+'\n');job.offset=0;job.rows.clear()
            old_cache=job.cache;old_bytes=old_cache.read_bytes();old_output=job.output
            document=job.cache.with_suffix('.review.json');old_document=document.read_bytes()
            self.assertTrue(job.snapshot()['outdated_instruction_parser'])
            with self.assertRaisesRegex(ValueError,'旧版未识别'):app.review(job.id)
            with patch.object(web,'read_saved_key',side_effect=AssertionError('no key')):
                app.retry(job.id);app.executor.submit(lambda:None).result(timeout=10)
            self.assertEqual(job.status,'done',job.error);self.assertTrue(job.dry_run)
            self.assertNotEqual(job.cache,old_cache);self.assertNotEqual(job.output,old_output)
            self.assertEqual(old_cache.read_bytes(),old_bytes);self.assertEqual(document.read_bytes(),old_document)
            self.assertFalse(job.snapshot()['outdated_instruction_parser'])
            self.assertEqual(job.id_summary['instruction_subject_alerts'],[])


if __name__=='__main__':unittest.main()
