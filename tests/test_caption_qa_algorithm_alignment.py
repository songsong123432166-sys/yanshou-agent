"""Current algorithm-format exceptions, number scopes and saved-review exports."""
import asyncio
import copy
import unittest

from captionqa import caption_qa_agent as qa
from captionqa import caption_qa_labels as labels
from captionqa import caption_qa_metadata as metadata
from captionqa.caption_field_schema import FIELDS, normalize, render
from captionqa.caption_qa_review import SafeAutoFixAgent, _review_records
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_summary import build_check_report
from tests.test_caption_qa_upgrades import GOOD

DUPLICATES={'duplicate_defined_subjects','duplicate_scene_numbers','duplicate_input_numbers'}


class AlgorithmAlignmentTests(unittest.TestCase):
    def duplicate_issues(self,sections):
        return [i for i in qa.run_static_rules(sections) if i['issue_type'] in DUPLICATES]

    def test_repeated_numbers_are_scoped_to_three_fields_and_media_types(self):
        sections={'主体':'<主体1>：男子。','场景':'<场景1>：房间。',
            '输入描述':'<图片1>：男子。\n<视频1>：房间。\n<音频1>：歌声。',
            '提示':'<图片1>、<图片1>。','画面元素':'<场景1>、<场景1>。',
            '总结':'<主体1>、<主体1>在<场景1>和<场景1>。',
            '自然指令':'<主体1>走动，<主体1>停下，<场景1>全景。'}
        self.assertEqual(self.duplicate_issues(sections),[])
        for field,extra,kind in [('主体','<主体1>','duplicate_defined_subjects'),
                ('场景','<场景1>','duplicate_scene_numbers'),
                ('输入描述','<图片1>','duplicate_input_numbers'),
                ('输入描述','<音频1>','duplicate_input_numbers')]:
            with self.subTest(field=field,extra=extra):
                items=self.duplicate_issues({**sections,field:sections[field]+extra+'。'})
                self.assertEqual([(i['section'],i['original_snippet'],i['issue_type']) for i in items],[(field,extra,kind)])
                self.assertFalse(items[0]['can_auto_fix'])

    def test_duplicate_occurrences_do_not_enlarge_subject_definition_set(self):
        sections={'主体':'<主体1>：站在<主体2>旁边的男子。\n<主体2>：女子。',
                  '自然指令':'<主体1>、<主体2>站立。'}
        self.assertEqual(qa.subject_coverage(sections)['defined_subjects'],['<主体1>','<主体2>'])
        self.assertEqual(self.duplicate_issues(sections)[0]['original_snippet'],'<主体2>')
        self.assertEqual(qa.apply_static_fixes(sections),sections)

    def test_formatting_keeps_prompt_chinese_titles_boolean_and_audio_literal(self):
        raw=GOOD+'## 任务类型\n人物参考 ｜ 场景参考\n## 提示\n参考<图片1>，生成短片。\n## similarity_label\nYes\n## 视觉标签\n实体/IP - 相同 ｜ 场景 - 相同\n'
        fixed=SafeAutoFixAgent(dry_run=True).check(raw)['revised_caption']
        self.assertIn('## 提示\n参考<图片1>，生成短片。',fixed)
        self.assertIn('## 参考关系标签\n是',fixed)
        self.assertIn('实体/ip-相同|场景-相同',fixed)
        self.assertIn('人物参考|场景参考',fixed)
        self.assertEqual(SafeAutoFixAgent(dry_run=True).check(fixed)['revised_caption'],fixed)
        literal='他说{实体/IP - 相同 ｜ 场景 - 相同}，画面显示【A | B】。'
        self.assertEqual(labels.normalize_separators('自然指令',literal),literal)
        self.assertEqual(metadata.normalize('video_similarity_tag','【实体/IP - 相同】'),'【实体/IP - 相同】')

    def test_time_ranges_still_have_blank_line_and_action_points_stay_inline(self):
        raw=GOOD.replace('0-4秒，第0秒，<主体1>向前跑。',
            '0 - 4秒，<主体1>走动。第2秒，停下。4-8秒，<主体1>转头。第6秒，镜头切换，挥手。')
        text=SafeAutoFixAgent(dry_run=True).check(raw)['revised_caption']
        self.assertIn('0 - 4秒，<主体1>走动。第2秒，停下。\n\n4-8秒，<主体1>转头。\n第6秒',text)
        self.assertEqual(SafeAutoFixAgent(dry_run=True).check(text)['revised_caption'],text)

    def test_one_click_fields_also_use_latest_metadata_format(self):
        values={name:'无。' for name in FIELDS}
        values.update(任务类型='人物参考',提示='生成短片。',similarity_label='是',video_similarity_tag='实体/ip-相同|场景-相同')
        expected=render(values)
        raw=expected.replace('实体/ip-相同|场景-相同','实体/IP - 相同 ｜ 场景 - 相同')
        result=asyncio.run(normalize(raw))
        self.assertEqual(result['caption'],expected)
        self.assertEqual(result['model_requests'],0)
        self.assertEqual(asyncio.run(normalize(expected))['status'],'already_standard')
        self.assertEqual(len(qa.parse_caption_sections(expected)),12)

    def test_old_approved_formats_are_validated_and_rendered_to_new_format(self):
        raw=GOOD+'## 视觉标签\nEntity/IP – Identical | Scene – Identical\n## 听觉标签\n音色 - 相同｜环境音 - 相同\n'
        record={'id':'sample-1','caption':raw,'qa_report':SafeAutoFixAgent(dry_run=True).check(raw)}
        doc=_review_records({1:record},caption_column='caption',id_column='id',source_sha256='synthetic')
        for field in doc['fields']:
            for issue in [i for i in doc['issues'] if i['field_id']==field['field_id']]:
                if issue['kind']=='pipe_width':issue['operations']=labels.previous_pipe_width_operations(field['name'],field['original'])
                if issue['kind']=='tag_language':issue['operations']=metadata.previous_operations(field['name'],field['original'])
        before=copy.deepcopy(doc)
        final=compile_review(doc,{1:record}).require_valid().rows[0].final_text
        self.assertIn('实体/ip-相同|场景-相同',final)
        self.assertIn('音色-相同|环境音-相同',final)
        self.assertEqual(doc,before)
        self.assertEqual(record['caption'],raw)

    def test_previous_v2_separator_policy_does_not_block_export(self):
        raw=GOOD+'## 听觉标签\n音色-相同，环境音-相同\n'
        record={'id':'1','caption':raw,'qa_report':SafeAutoFixAgent(dry_run=True).check(raw)}
        doc=_review_records({1:record},caption_column='caption',id_column='id',source_sha256='synthetic')
        field=next(f for f in doc['fields'] if f['name']=='audio_similarity_tag')
        item=next(i for i in doc['issues'] if i['kind']=='label_separator')
        item.update(separator_policy=labels.PREVIOUS_POLICY,operations=labels.previous_separator_operations(field['name'],field['original']))
        self.assertIn('音色-相同|环境音-相同',compile_review(doc,{1:record}).require_valid().rows[0].final_text)
        item['operations'][0]['new']='音乐-不同'
        self.assertTrue(compile_review(doc,{1:record}).blockers)

    def test_new_duplicate_findings_have_readable_review_and_summary_labels(self):
        raw=GOOD+'## 场景\n<场景1>：房间。\n<场景1>：走廊。\n## 输入描述\n<图片1>：男子。\n<图片1>：女子。\n'
        report=SafeAutoFixAgent(dry_run=True).check(raw)
        record={'id':'1','caption':raw,'qa_report':report}
        doc=_review_records({1:record},caption_column='caption',id_column='id',source_sha256='synthetic')
        self.assertEqual({i['label'] for i in doc['issues'] if i['kind'] in DUPLICATES},{'场景编号重复','输入素材编号重复'})
        self.assertTrue(all(not i['operations'] and i['decision']=='pending' for i in doc['issues'] if i['kind'] in DUPLICATES))
        items=build_check_report([{'row_index':1,'qa_report':report}])['items']
        self.assertEqual({i['key'] for i in items if i['review']},{'duplicate_scenes','duplicate_inputs'})
        self.assertEqual(report['revised_caption'],raw)


if __name__=='__main__':unittest.main()
