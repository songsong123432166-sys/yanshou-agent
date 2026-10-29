"""Portable observable invariants; no model, network, project import or secret."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SKILL/'scripts'))
import caption_skill as app
import _schema as schema


def caption(instruction='<主体1>转头，面向镜头。'):
    fields={'任务类型':'人物参考|场景参考','提示':'生成一个人物转头的视频。','similarity_label':'yes',
        'video_similarity_tag':'实体-相同|场景-相同','audio_similarity_tag':'无',
        '输入描述':'<图片1>：一名男子。','基调':'写实。','场景':'<场景1>：室内看台。',
        '主体':'<主体1>：<图片1>中的男子，身穿白色衬衫。','画面元素':'无。',
        '总结':'<主体1>在<场景1>中转头。','自然指令':instruction}
    return schema.render(fields)


def suggestion(report,original,replacement,label='疑似病句',field='自然指令',**extra):
    return {'base_sha256':report['base_sha256'],'semantic_status':'completed','findings':[
        {'field':field,'label':label,'original':original,'replacement':replacement,
         'reason':'明确的局部表达调整，保留原有事实。',**extra}]}


class LocalTests(unittest.TestCase):
    def test_field_spacing_is_safe_and_check_only_keeps_original(self):
        expected=caption('<主体1>说{hello\n## subject\nworld}。\n\n随后转身。')
        raw=expected.replace('\n\n## ', '\n## ').rstrip('\n')
        report=app.check(raw)
        self.assertEqual(report['caption'],expected)
        self.assertTrue(report['automatic_changes'])
        self.assertEqual(app.check(raw,check_only=True)['caption'],raw)

    def test_good_caption_is_unchanged_and_semantics_not_claimed(self):
        raw=caption();report=app.check(raw)
        self.assertEqual(report['caption'],raw);self.assertEqual(report['findings'],[])
        self.assertEqual(report['semantic_status'],'not_run');self.assertIsNone(report['sample_id'])

    def test_header_spacing_aliases_and_metadata_values(self):
        raw=caption().replace('## 任务类型','##任务类型').replace('## video_similarity_tag','similarity_tag')
        report=app.check(raw)
        self.assertEqual(report['caption'],caption());self.assertTrue(report['automatic_changes'])

    def test_natural_description_alias_reads_subjects_before_reporting_missing(self):
        raw=caption().replace('## 自然指令','## 自然描述')
        report=app.check(raw)
        self.assertEqual(report['caption'],caption())
        self.assertFalse(any(i['label']=='自然指令漏写已定义主体' for i in report['findings']))

    def test_every_heading_has_exactly_one_ascii_space(self):
        for prefix in ('##','##  ','##\t','##\u3000'):
            report=app.check(caption().replace('## 任务类型',prefix+'任务类型'))
            self.assertEqual(report['caption'],caption())

    def test_tag_period_and_dictionary_typo(self):
        raw=caption('主体1睁开眼晴')
        report=app.check(raw)
        self.assertIn('<主体1>睁开眼睛。',report['caption']);self.assertFalse(report['findings'])

    def test_midparagraph_speech_kept_and_outer_period_added_only_at_end(self):
        raw=caption('<主体1>说道{你好}，然后说道{私は異世界に来たようだ。}')
        fixed=app.check(raw)['caption']
        self.assertIn('说道{你好}，然后',fixed);self.assertIn('{私は異世界に来たようだ。}。',fixed)

    def test_audio_and_screen_interiors_are_byte_identical(self):
        for literal in ['{主体2, 眼晴}', '（背景音乐是《You(=I)》, 钢琴眼晴, 主体2）',
                        '<环境有眼晴, 主体2的呜呜声>', '【眼晴 主体2, 99】']:
            with self.subTest(literal=literal):
                fixed=app.check(caption('<主体1>站立，'+literal))['caption']
                self.assertIn(literal+'。',fixed)

    def test_outer_audio_wrappers_can_change_without_touching_interior(self):
        for old,new in [('｛Hello, 眼晴｝','{Hello, 眼晴}'),
                        ('(背景音乐是《You(=I)》, 眼晴)','（背景音乐是《You(=I)》, 眼晴）'),
                        ('＜传来噼啪, 的燃烧声＞','<传来噼啪, 的燃烧声>')]:
            self.assertIn(new+'。',app.check(caption('<主体1>站立。'+old))['caption'])

    def test_multiline_and_unclosed_literals_are_not_punctuated_inside(self):
        for literal in ['{hello\n主体2 眼晴\nworld}', '【hello\n主体2 眼晴\nworld】']:
            self.assertIn(literal+'。',app.check(caption('<主体1>说道'+literal))['caption'])
        for literal in ['{hello\n眼晴','【hello\n眼晴','（背景音乐是眼晴, 钢琴']:
            fixed=app.check(caption('<主体1>站立。'+literal))['caption']
            self.assertIn(literal,fixed);self.assertNotIn('眼睛',fixed)

    def test_urls_timecodes_numbers_and_english_are_kept(self):
        text='<主体1>查看https://example.com/a?q=1.5，时间01:02，第1.5秒。Hello, world!'
        self.assertIn(text,app.check(caption(text))['caption'])

    def test_crlf_blank_lines_and_body_spaces_preserved(self):
        raw=caption('<主体1>站立。第二句。  \n\n  第2秒，转身。').replace('\n','\r\n')
        self.assertEqual(app.check(raw)['caption'],raw)

    def test_disabled_items_are_neither_changed_nor_reported(self):
        raw=caption('<主体1>：<主体1>：缓缓缓缓转身，穿着红色长袍。')
        report=app.check(raw)
        self.assertEqual(report['caption'],raw);self.assertEqual(report['findings'],[])

    def test_grid_task_and_scene_definitions_never_renumbered(self):
        raw=caption().replace('人物参考|场景参考','九宫格/少宫格').replace(
            '<场景1>：室内看台。','<场景2>：左侧看台。\n<场景2>：右侧大厅。').replace('## 场景','##场景')
        report=app.check(raw)
        self.assertEqual(report['caption'],raw.replace('##场景','## 场景'))

    def test_missing_picture_elements_marked_and_added_empty(self):
        raw=caption().replace('## 画面元素\n无。\n\n','')
        report=app.check(raw,'0007')
        self.assertEqual(report['missing_fields'],['画面元素']);self.assertEqual(report['sample_id'],'0007')
        self.assertIn('## 画面元素\n\n',report['caption']);self.assertNotIn('## 画面元素\n无',report['caption'])
        self.assertTrue(any(i['label']=='缺少字段' for i in report['findings']))

    def test_missing_subject_and_ghosts_are_pending_not_auto_added(self):
        raw=caption('<主体9>站立。')
        report=app.check(raw)
        labels={i['label'] for i in report['findings']}
        self.assertIn('自然指令漏写已定义主体',labels);self.assertIn('引用未定义主体',labels)
        self.assertEqual(report['caption'],raw)
        self.assertTrue(all(i.get('subject') for i in report['findings']))

    def test_descriptive_subject_reference_is_not_a_duplicate_definition(self):
        raw=caption().replace('<图片1>中的男子，身穿白色衬衫。','音色同<主体1>，身穿白色衬衫。')
        self.assertFalse(any(i['label']=='主体重复定义' for i in app.check(raw)['findings']))

    def test_old_structure_unknown_header_or_duplicate_is_preserved(self):
        for raw in ['## task_type\n参考\n## natural_instruction\n生成视频。',
                    caption()+'## 未知标题\n不要丢掉。',caption()+'## 主体\n<主体2>：另一人。']:
            report=app.check(raw)
            self.assertEqual(report['caption'],raw);self.assertEqual(report['status'],'STRUCTURE_NEEDS_CONFIRMATION')

    def test_check_only_never_changes_input(self):
        raw=caption('主体1睁开眼晴')
        report=app.check(raw,check_only=True)
        self.assertEqual(report['caption'],raw);self.assertTrue(report['findings'])
        self.assertEqual(report['automatic_changes'],[]);self.assertTrue(report['proposed_safe_changes'])


class ConfirmationTests(unittest.TestCase):
    def setUp(self):self.raw=caption('<主体1>抬起手看向镜头。');self.report=app.check(self.raw)

    def plan(self):return app.propose(self.report,suggestion(self.report,'抬起手看向镜头。','抬起手，随后看向镜头。'))

    def test_proposal_does_not_apply_until_explicit_confirmation(self):
        plan=self.plan();self.assertEqual(plan['caption'],self.raw)
        with self.assertRaises(ValueError):app.apply(plan,['Q001'],'')
        result=app.apply(plan,['Q001'],'确认按Q001的建议修改')
        self.assertIn('抬起手，随后看向镜头。',result['caption']);self.assertEqual(len(result['approved_history']),1)
        self.assertEqual(result['semantic_status'],'needs_recheck')

    def test_only_selected_item_applies_and_other_stays_pending(self):
        data=suggestion(self.report,'抬起手','举起手')
        data['findings'].append({'field':'自然指令','label':'疑似病句','original':'看向镜头。','replacement':'面向镜头。','reason':'局部候选供人工核对。'})
        plan=app.propose(self.report,data);result=app.apply(plan,['Q001'],'仅确认Q001')
        self.assertIn('举起手看向镜头。',result['caption'])
        self.assertTrue(any(i.get('replacement')=='面向镜头。' for i in result['findings']))
        self.assertEqual(next(i['issue_id'] for i in result['findings'] if i.get('replacement')=='面向镜头。'),'Q002')
        with self.assertRaises(ValueError):app.apply(result,['Q001'],'旧编号不能套用')

    def test_stale_digest_missing_position_and_overlaps_are_rejected(self):
        data=suggestion(self.report,'抬起手','举起手');data['base_sha256']='stale'
        with self.assertRaises(ValueError):app.propose(self.report,data)
        with self.assertRaises(ValueError):app.propose(self.report,suggestion(self.report,'不存在','新文本'))
        data=suggestion(self.report,'抬起手','举起手')
        data['findings'].append({'field':'自然指令','label':'疑似病句','original':'手看向','replacement':'手，转向','reason':'另一条候选。'})
        plan=app.propose(self.report,data)
        with self.assertRaisesRegex(ValueError,'重叠'):app.apply(plan,['Q001','Q002'],'确认两项')

    def test_repeated_snippet_requires_unambiguous_occurrence(self):
        report=app.check(caption('<主体1>抬手。然后抬手。'))
        data=suggestion(report,'抬手','举手')
        with self.assertRaisesRegex(ValueError,'多处'):app.propose(report,data)
        data['findings'][0]['occurrence']=2
        result=app.apply(app.propose(report,data),['Q001'],'仅第二处')
        self.assertIn('<主体1>抬手。然后举手。',result['caption'])

    def test_numbers_scenes_materials_screens_and_boundaries_are_protected(self):
        raw=caption('<主体1>在第2秒查看<图片1>，<场景1>中显示【hello】。')
        report=app.check(raw)
        for old,new in [('第2秒','第3秒'),('<图片1>','<图片2>'),('<场景1>','<场景2>'),('【hello】','【bye】'),
                        ('查看','查看\n## 主体\n')]:
            with self.subTest(old=old,new=new):
                with self.assertRaises(ValueError):app.propose(report,suggestion(report,old,new))

    def test_subject_reference_requires_existing_definition_and_confirmation(self):
        report=app.check(caption('<主体9>转头。'))
        data=suggestion(report,'<主体9>','<主体1>',label='主体引用')
        plan=app.propose(report,data);identifier=next(i['issue_id'] for i in plan['findings'] if i.get('replacement'))
        result=app.apply(plan,[identifier],'核对后确认主体1')
        self.assertIn('<主体1>转头。',result['caption'])
        with self.assertRaises(ValueError):app.propose(report,suggestion(report,'<主体9>','<主体8>',label='主体引用'))

    def test_audio_punctuation_requires_separate_explicit_approval_and_words_stay_protected(self):
        report=app.check(caption('<主体1>说道{Now, excuse me，}。'))
        data=suggestion(report,'me，','me,',label='标点问题')
        plan=app.propose(report,data)
        self.assertTrue(plan['findings'][0]['operation']['audio_confirmation_required'])
        with self.assertRaises(ValueError):app.apply(plan,['Q001'],'确认Q001')
        result=app.apply(plan,['Q001'],'已核对音频，确认该标点',allow_audio=True)
        self.assertIn('{Now, excuse me,}',result['caption'])
        with self.assertRaises(ValueError):app.propose(report,suggestion(report,'excuse','forgive',label='疑似错别字'))

    def test_disabled_categories_cannot_be_reintroduced(self):
        for label,reason in [('属性冲突','两字段颜色不同'),('疑似病句','属性冲突，需要统一服饰')]:
            data=suggestion(self.report,'抬起手','举起手',label=label);data['findings'][0]['reason']=reason
            with self.assertRaises(ValueError):app.propose(self.report,data)

    def test_metadata_meanings_cannot_be_guessed_by_a_candidate(self):
        with self.assertRaises(ValueError):
            app.propose(self.report,suggestion(self.report,'人物参考','场景参考',label='内容符号',field='任务类型'))

    def test_no_candidate_does_not_enable_apply(self):
        plan=app.propose(self.report,suggestion(self.report,'抬起手',None))
        with self.assertRaises(ValueError):app.apply(plan,['Q001'],'确认')

    def test_semantic_completion_and_failure_are_distinct(self):
        for status,expected in [('completed','SCREENING_NO_PENDING'),('incomplete','SEMANTIC_INCOMPLETE')]:
            result=app.propose(self.report,{'base_sha256':self.report['base_sha256'],'semantic_status':status,'findings':[]})
            self.assertEqual(result['status'],expected)


class DeliveryTests(unittest.TestCase):
    def test_cli_is_portable_preserves_source_and_creates_new_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'原始 caption.txt';raw=caption('主体1睁开眼晴');source.write_text(raw,encoding='utf-8')
            command=[sys.executable,str(SKILL/'scripts/caption_skill.py'),'check','--input',str(source),'--out-dir',str(root/'results')]
            paths=[]
            for _ in range(2):
                proc=subprocess.run(command,cwd=root,capture_output=True,text=True,check=True)
                result=json.loads(proc.stdout);paths.append(result['caption'])
                self.assertIn('<主体1>睁开眼睛。',Path(result['caption']).read_text())
            self.assertNotEqual(paths[0],paths[1]);self.assertEqual(source.read_text(),raw)

    def test_tampered_reports_and_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';report=app.check(caption());report['caption']+='changed'
            path.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):app.read_report(path)
            path.write_text('{"key":1,"key":2}')
            with self.assertRaises(ValueError):app.json_file(path)


if __name__=='__main__':unittest.main()
