import asyncio
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from captionqa import caption_qa_agent as qa
from captionqa.caption_field_schema import FIELDS, render, normalize, parse
from captionqa.caption_qa_review import SafeAutoFixAgent, IdentificationAgent, _review_records
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_review_io import export_approved
from captionqa.caption_qa_contract import enrich
from captionqa.caption_qa_metadata import normalize as translate, unknown_english
from captionqa.caption_qa_references import count_cell, detect_columns, findings
from tests.test_caption_qa_upgrades import GOOD

TAGS = '\n##Similarity Label\nYes\n##  Video Similarity Tag\nEntity/IP – Identical | VFX – Identical\n##Audio Similarity Tag\nSound Effect – Same Category, Different Instance\n'


class MetadataTests(unittest.TestCase):
    def test_screenshot_tags_are_chinese_with_relations_preserved(self):
        result = SafeAutoFixAgent(dry_run=True).check(GOOD+TAGS)
        text = result['revised_caption']
        self.assertIn('## 参考关系标签\n是', text)
        self.assertIn('## 视觉标签\n实体/IP - 相同｜特效 - 相同', text)
        self.assertIn('## 听觉标签\n音效 - 同类不同个', text)
        self.assertEqual(result['status'], 'AUTO_FIXED')
        self.assertFalse(result['need_human_review'])
        for line in text.splitlines():
            if line.startswith('##'): self.assertRegex(line, r'^## [^ ]')
        self.assertEqual(SafeAutoFixAgent(dry_run=True).check(text)['revised_caption'], text)

    def test_boolean_labels_translate_case_insensitively_without_touching_body(self):
        for value,expected in [('Yes','是'),('YES','是'),('yes','是'),('No','否'),('NO','否'),('no','否')]:
            self.assertEqual(translate('similarity_label','  '+value+'\n'),'  '+expected+'\n')
        misplaced=SafeAutoFixAgent(dry_run=True).check(GOOD+'\n## video_similarity_tag\nYes\n')
        self.assertIn('## 视觉标签\n是',misplaced['revised_caption'])
        self.assertIn('字段归属',misplaced['human_review_reason'])
        self.assertEqual(translate('自然指令','Yes/No。'),'Yes/No。')
        self.assertEqual(translate('similarity_label','Yes/No'),'Yes/No')

    def test_english_prose_and_unknown_labels_are_not_guessed(self):
        self.assertEqual(translate('自然指令', 'VFX – Identical'), 'VFX – Identical')
        self.assertEqual(translate('video_similarity_tag', 'The scene is identical'), 'The scene is identical')
        result = SafeAutoFixAgent(dry_run=True).check(GOOD+'\n## 视觉标签\nUnknown Label\n')
        self.assertIn('Unknown Label', result['revised_caption'])
        self.assertTrue(result['need_human_review'])
        self.assertFalse(unknown_english('video_similarity_tag', '实体IP - 相同'))

    def test_new_and_legacy_titles_parse_to_stable_field_keys(self):
        for heading in ('video_similarity_tag', 'similarity_tag', 'Video Similarity Tag', '视觉标签'):
            self.assertEqual(qa.parse_caption_sections('##'+heading+'\n实体/IP\n')['video_similarity_tag'], '实体/IP\n')
        result = asyncio.run(normalize(render({n:'人物参考' if n=='任务类型' else '。' for n in FIELDS})
            .replace('## 视觉标签','##Video Similarity Tag').replace('## 听觉标签','##audio_similarity_tag')))
        self.assertEqual(result['status'], 'converted')
        self.assertIn('## 视觉标签', result['caption'])
        self.assertEqual(asyncio.run(normalize(result['caption']))['status'],'already_standard')

    def test_inline_headings_and_quoted_words_preserve_bodies(self):
        text = GOOD.replace('## 主体\n', '##主体  ')+TAGS
        result = SafeAutoFixAgent(dry_run=True).check(text)['revised_caption']
        self.assertIn('## 主体  ', result)
        self.assertEqual(qa.parse_caption_sections(result)['主体'],qa.parse_caption_sections(text)['主体'])
        self.assertEqual(translate('audio_similarity_tag','【Sound Effect】'), '【Sound Effect】')


class ReferenceTests(unittest.TestCase):
    def test_five_images_disallow_six_zero_and_bare_six_but_allow_reuse(self):
        row={'ref_images':[f'image{i}.jpg' for i in range(5)],'ref_videos':['video1.mp4'],'ref_audios':[]}
        columns=detect_columns([row])
        issues, counts, _=findings({'自然指令':'参考<图片5>、<图片5>、<图片6>，以及参考图片0和<视频2>。'},row,columns)
        self.assertEqual(counts,{'图片':5,'视频':1,'音频':0})
        self.assertEqual({i['original_snippet'] for i in issues},{'图片6','图片0','视频2'})
        self.assertTrue(all(not i['can_auto_fix'] for i in issues))
        self.assertIn('5 个 ref 图片', next(i['suggestion'] for i in issues if i['original_snippet']=='图片6'))

    def test_grouped_headers_count_each_media_and_exclude_target(self):
        headers={'样本ID':'','素材数':'图/视/音','参考素材':'图片','__column_7':'音频','__column_8':'视频','目标视频':''}
        row={'样本ID':'x','素材数':'2/1/0','参考素材':'https://x/1.jpg\nhttps://x/2.png',
             '__column_7':'','__column_8':'https://x/1.mp4','目标视频':'https://x/target.mp4'}
        columns=detect_columns([headers,row])
        issues,counts,notes=findings({'输入描述':'<图片3>、<视频2>和<音频1>。'},row,columns)
        self.assertEqual(counts,{'图片':2,'视频':1,'音频':0})
        self.assertEqual(len(issues),3);self.assertEqual(notes,[])

    def test_source_count_mismatch_uses_actual_attachments(self):
        headers={'素材数':'图/视/音','参考素材':'图片','__column_7':'音频','__column_8':'视频'}
        row={'素材数':'5/0/0','参考素材':'a.jpg\nb.png','__column_7':'','__column_8':''}
        issues,counts,notes=findings({'输入描述':'<图片3>。'},row,detect_columns([headers,row]))
        self.assertEqual(counts['图片'],2);self.assertEqual(len(issues),1)
        self.assertIn('实际参考素材列为 2', notes[0])

    def test_unknown_inventory_is_not_zero_or_guessed_from_caption(self):
        row={'ref_images':'附件内容待确认'}
        issues,counts,notes=findings({'输入描述':'<图片6>，共有6张图片。'},row,detect_columns([row]))
        self.assertEqual(issues,[]);self.assertIsNone(counts['图片']);self.assertTrue(notes)
        self.assertIsNone(count_cell('a.jpg and some notes'))

    def test_attachment_json_and_urls_are_local_only(self):
        self.assertEqual(count_cell('[{"url":"https://x/1.jpg"},{"name":"2.png"}]'),2)
        self.assertEqual(count_cell('https://x/1.jpg,https://x/2.png'),2)
        self.assertEqual(count_cell([]),0)

    def test_literal_speech_and_screen_text_are_not_media_references(self):
        row={'ref_images':[]}
        issues,_,_=findings({'自然指令':'他说{图片6}，屏幕显示【图片6】；网址https://x/图片6.jpg 。'},row,detect_columns([row]))
        self.assertEqual(issues,[])

    def test_csv_batch_report_and_review_export_keep_source_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.csv';cache=root/'cache.jsonl'
            text=GOOD.replace('<主体1>奔跑。','<主体1>参考<图片6>奔跑。')+TAGS
            with source.open('w',encoding='utf-8-sig',newline='') as stream:
                w=csv.writer(stream);w.writerow(['id','ref_images','目标视频','caption'])
                w.writerow(['0001','\n'.join(f'https://x/{i}.jpg' for i in range(5)),'target.mp4',text])
            asyncio.run(qa.aprocess_batch(source,cache,SafeAutoFixAgent(dry_run=True),column='caption'))
            row=json.loads(cache.read_text())
            self.assertEqual(row['qa_status'],'NEED_HUMAN_REVIEW')
            self.assertIn('图片6',row['qa_human_review_reason'])
            self.assertEqual(row['qa_report']['reference_validation']['counts']['图片'],5)
            document=_review_records({1:row},caption_column='caption',id_column='id',source_sha256=qa._source_sha256(source))
            document.update(input_sha256=qa._source_sha256(source),cache_sha256=qa._source_sha256(cache),sheet=None)
            plan=compile_review(document,{1:row});plan.require_valid()
            output=export_approved(document,source,source,cache,root,'source',plan=plan)
            with Path(output['corrected_path']).open(encoding='utf-8-sig',newline='') as stream: exported=list(csv.reader(stream))
            self.assertEqual(exported[1][:3],['0001',row['ref_images'],'target.mp4'])
            self.assertIn('<图片6>',exported[1][3]);self.assertIn('## 视觉标签',exported[1][3])
            self.assertTrue(any(i['kind']=='reference_out_of_range' and i['decision']=='pending' for i in document['issues']))

    def test_new_bad_reference_in_manual_draft_is_rechecked(self):
        text=GOOD.replace('<主体1>奔跑。','<主体1>奔跑')
        document=_review_records({1:{'id':'1','caption':text,'qa_report':SafeAutoFixAgent(dry_run=True).check(text)}},
            caption_column='caption',id_column='id',source_sha256='source')
        field=next(f for f in document['fields'] if f['name']=='总结')
        document['field_edits'][field['field_id']]='<主体1>参考<图片6>奔跑。'
        records={1:{'id':'1','caption':text,'ref_images':['1.jpg']}}
        enrich(document,records)
        plan=compile_review(document,records)
        self.assertTrue(any('图片6' in w['message'] for w in plan.review_checks))
        self.assertIn('<图片6>',plan.captions[1])

    def test_review_migration_keeps_manual_edits_rejects_and_is_idempotent(self):
        text=GOOD+TAGS
        document=_review_records({1:{'id':'1','caption':text,'qa_report':IdentificationAgent(dry_run=True).check(text)}},
            caption_column='caption',id_column='id',source_sha256='source')
        field=next(f for f in document['fields'] if f['name']=='video_similarity_tag')
        document['field_edits'][field['field_id']]='人工保留的视觉标签'
        issue=next(i for i in document['issues'] if i['field_id']==field['field_id']);issue['decision']='reject'
        before=copy.deepcopy(document)
        records={1:{'id':'1','caption':text,'ref_images':[]}}
        enrich(document,records)
        self.assertEqual(document['field_edits'],before['field_edits'])
        self.assertEqual(document['issues'][:len(before['issues'])],before['issues'])
        self.assertEqual(enrich(document,records),[])
        result=compile_review(document,records).require_valid()
        self.assertIn('人工保留的视觉标签',result.captions[1])


if __name__=='__main__': unittest.main()
