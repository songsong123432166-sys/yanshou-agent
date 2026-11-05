import tempfile
import unittest
from pathlib import Path
import openpyxl
from captionqa.caption_field_jobs import detect_column,select_table

CAPTION='## 任务类型\n场景参考\n\n## 自然指令\n0-1秒，镜头后移。'

class FieldSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def save(self,rows,extra=None):
        book=openpyxl.Workbook();sheet=book.active;sheet.title='数据'
        for row in rows:sheet.append(row)
        if extra:
            for name,data in extra.items():
                page=book.create_sheet(name)
                for row in data:page.append(row)
        path=self.root/'input.xlsx';book.save(path);book.close();return path
    def test_group_header_flag_does_not_outrank_real_second_row_caption(self):
        source=self.save([['样本ID','人工标注',None,None,None,'机器标注参考',None],
                          ['', '是否运镜','运镜类型','运镜描述','caption','运镜类型','机标caption'],
                          ['001','是','后移','持续后移',CAPTION,'后移',CAPTION]])
        self.assertEqual(detect_column(source,'数据'),'__column_5')
        self.assertEqual(select_table(source),('数据','__column_5'))
    def test_populated_human_caption_keeps_priority_over_generic_caption(self):
        source=self.save([['样本ID','人工标注','caption'],['001',CAPTION,CAPTION]])
        self.assertEqual(detect_column(source,'数据'),'人工标注')
    def test_machine_and_notes_are_never_content_fallback(self):
        source=self.save([['样本ID','人工标注','机标caption','备注'],['001','是',CAPTION,CAPTION]])
        with self.assertRaisesRegex(ValueError,'未找到含可识别 Caption'):
            select_table(source)
    def test_flat_human_column_does_not_silently_fall_back_to_other_caption(self):
        source=self.save([['样本ID','人工标注','caption'],['001','尚未整理的人工原文',CAPTION]])
        self.assertEqual(detect_column(source,'数据'),'人工标注')
        with self.assertRaisesRegex(ValueError,'未找到含可识别 Caption'):
            select_table(source)
    def test_ambiguous_caption_columns_still_require_explicit_column(self):
        source=self.save([['样本ID','caption A','caption B'],['001',CAPTION,CAPTION]])
        with self.assertRaisesRegex(ValueError,'唯一的Caption'):
            detect_column(source,'数据')
        self.assertEqual(detect_column(source,'数据','caption B'),'caption B')
    def test_multiple_data_sheets_are_named_in_error(self):
        rows=[['样本ID','caption'],['001',CAPTION]]
        source=self.save(rows,{'其他数据':rows})
        with self.assertRaisesRegex(ValueError,'发现多个.*数据.*其他数据'):
            select_table(source)
        self.assertEqual(select_table(source,sheet='其他数据'),('其他数据','caption'))
    def test_reference_sheet_does_not_override_main_data_sheet(self):
        rows=[['样本ID','caption'],['001',CAPTION]]
        self.assertEqual(select_table(self.save(rows,{'参考示例':rows})),('数据','caption'))
    def test_empty_table_reports_missing_caption_not_multiple_sheets(self):
        source=self.save([['样本ID','caption'],['001','']])
        with self.assertRaisesRegex(ValueError,'未找到含可识别 Caption'):
            select_table(source)

if __name__=='__main__':unittest.main()
