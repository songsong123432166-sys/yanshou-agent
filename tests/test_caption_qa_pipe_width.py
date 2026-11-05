"""Fullwidth label pipes, legacy confirmations and human-draft export parity."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from captionqa import caption_qa_agent as qa
from captionqa.caption_qa_labels import pipe_width_operations,normalize_pipe_width,legacy_separator_operations,LEGACY_POLICY
from captionqa.caption_qa_review import SafeAutoFixAgent,IdentificationAgent,make_review,decide
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_store import ReviewStore,source_records
from captionqa.caption_qa_review_io import export_approved
from tests.test_caption_qa_upgrades import GOOD

class PipeWidthTests(unittest.TestCase):
    def test_pipe_has_no_surrounding_spaces_and_only_metadata_hyphens_gain_spaces(self):
        value='实体/IP-相同 | 场景 -相同　｜　视觉风格- 相同'
        expected='实体/IP - 相同｜场景 - 相同｜视觉风格 - 相同'
        self.assertEqual(normalize_pipe_width('video_similarity_tag',value),expected)
        self.assertEqual(normalize_pipe_width('video_similarity_tag',expected),expected)
        for text in ['0-8秒，<主体1>站立。','0 - 8秒，<主体1>站立。','他说{A | B，0-8秒}，显示【A | B】。']:
            self.assertEqual(normalize_pipe_width('自然指令',text),text)

    def test_only_label_separators_change_and_opaque_pipes_survive(self):
        text='实体/IP-相同|未知标签|{A|B}|【C|D】|“E|F”|https://a.test/x|y'
        expected='实体/IP - 相同｜未知标签｜{A|B}｜【C|D】｜“E|F”｜https://a.test/x|y'
        self.assertEqual(normalize_pipe_width('video_similarity_tag',text),expected)
        self.assertEqual(normalize_pipe_width('video_similarity_tag',expected),expected)
        self.assertEqual(normalize_pipe_width('自然指令',text),text)
        self.assertEqual(normalize_pipe_width('audio_similarity_tag','A||B'),'A｜｜B')
        self.assertEqual(pipe_width_operations('task_type','A|B\n## 提示\n正文'),[])

    def test_default_automatic_flow_preserves_literal_pipes_in_the_caption(self):
        raw=GOOD.replace('<主体1>向前跑。','<主体1>说道{A|B}，展示【C|D】。')+'## 任务类型\n人物参考|场景参考\n'
        report=SafeAutoFixAgent(dry_run=True).check(raw)
        self.assertEqual(report['revised_caption'],raw.replace('人物参考|场景参考','人物参考｜场景参考'))
        self.assertTrue(next(i for i in report['issues'] if i['issue_type']=='pipe_width')['resolved'])


class PipeExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'input.jsonl';self.cache=self.root/'checkpoint.jsonl'

    def build(self,value='音色-相同|环境音-相同',automatic=True):
        self.caption=GOOD+'## audio_similarity_tag\n'+value+'\n'
        self.source.write_text(json.dumps({'id':'0001','caption':self.caption,'note':'A|B'})+'\n')
        agent=(SafeAutoFixAgent if automatic else IdentificationAgent)(dry_run=True)
        asyncio.run(qa.aprocess_batch(self.source,self.cache,agent,column='caption'))
        return make_review(self.cache,self.source,self.source,caption_column='caption',id_column='id')

    def plan(self,doc):return compile_review(doc,source_records(doc,self.source))

    def test_review_all_keeps_the_original_until_the_width_fix_is_approved(self):
        doc=self.build(automatic=False);issue=next(i for i in doc['issues'] if i['kind']=='pipe_width')
        self.assertIn('音色-相同|环境音-相同',self.plan(doc).rows[0].final_text)
        self.assertIn('## 听觉标签',self.plan(doc).rows[0].final_text)
        doc=decide(doc,revision=0,issue_id=issue['issue_id'],decision='approve')
        self.assertIn('音色 - 相同｜环境音 - 相同',self.plan(doc).rows[0].final_text)

    def test_old_approved_separator_operations_remain_valid_and_output_fullwidth(self):
        doc=self.build('音色-相同|环境音-相同，音乐-相同')
        label=next(i for i in doc['issues'] if i['kind']=='label_separator');field=doc['fields'][0]
        label.update(decision='approve',separator_policy=LEGACY_POLICY,
                     operations=legacy_separator_operations(field['name'],field['original']))
        doc['issues']=[label];field['issue_ids']=[label['issue_id']]
        before=copy.deepcopy(doc);plan=self.plan(doc).require_valid()
        self.assertIn('音色 - 相同｜环境音 - 相同｜音乐 - 相同',plan.rows[0].final_text)
        self.assertEqual(doc,before)

    def test_saved_manual_words_are_retained_and_preview_matches_export(self):
        doc=self.build('音色-相同，环境音-相同');field=doc['fields'][0]
        manual='音色-不同|环境音-相同'
        doc=decide(doc,revision=0,field_id=field['field_id'],final_text=manual)
        before=copy.deepcopy(doc);plan=self.plan(doc).require_valid()
        self.assertEqual(plan.fields[0].editor_text,'音色 - 不同｜环境音 - 相同')
        result=export_approved(doc,self.source,self.source,self.cache,self.root/'out','sample',plan=plan)
        out=json.loads(Path(result['corrected_path']).read_text())
        self.assertIn('音色 - 不同｜环境音 - 相同',out['caption']);self.assertEqual(out['note'],'A|B')
        self.assertEqual(doc,before);self.assertEqual(doc['field_edits'][field['field_id']],manual)
        self.assertEqual(json.loads(self.source.read_text())['caption'],self.caption)

    def test_old_pass_review_upgrades_once_without_overwriting_the_source(self):
        doc=self.build();doc['fields']=[];doc['issues']=[];doc['events']=[]
        store=ReviewStore(self.source,self.source,self.cache);store.save(doc);source=self.source.read_bytes();cache=self.cache.read_bytes()
        updated=store.load();self.assertEqual(updated['revision'],1)
        self.assertEqual(updated['issues'][0]['decision'],'auto')
        self.assertEqual(store.load(),updated)
        self.assertIn('音色 - 相同｜环境音 - 相同',store.plan(updated).rows[0].final_text)
        self.assertEqual(self.source.read_bytes(),source);self.assertEqual(self.cache.read_bytes(),cache)

if __name__=='__main__':unittest.main()
