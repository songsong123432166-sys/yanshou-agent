"""Split workbook scope/identity and web API-mode tests; no real API calls."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZipFile

import caption_qa_agent as qa
import caption_qa_web as web
from caption_qa_split_input import FIELDS, HEADERS, prepare_split_input
from test_caption_qa_upgrades import FakeAsyncClient, request_sections, semantic_reply


def workbook(path, sheets):
    """Small OOXML fixture built without spreadsheet authoring dependencies."""
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    package = 'http://schemas.openxmlformats.org/package/2006/relationships'
    content = 'http://schemas.openxmlformats.org/package/2006/content-types'
    with ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', f'<Types xmlns="{content}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1)) + '</Types>')
        z.writestr('_rels/.rels', f'<Relationships xmlns="{package}"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>' + ''.join(f'<sheet name={quoteattr(name)} sheetId="{i}" r:id="rId{i}"/>' for i,(name,_) in enumerate(sheets,1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{package}">' + ''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1)) + '</Relationships>')
        for i,(_,rows) in enumerate(sheets,1):
            body=[]
            for n,row in enumerate(rows,1):
                cells=[]
                for j,value in enumerate(row):
                    if value is None:
                        continue
                    coord=f'{chr(65+j)}{n}'
                    if isinstance(value,tuple):
                        cells.append(f'<c r="{coord}"><f>{escape(value[0])}</f></c>')
                    elif isinstance(value,(int,float)):
                        cells.append(f'<c r="{coord}" t="n"><v>{value}</v></c>')
                    else:
                        cells.append(f'<c r="{coord}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
                body.append(f'<row r="{n}">' + ''.join(cells) + '</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{main}"><dimension ref="A1"/><sheetData>' + ''.join(body) + '</sheetData></worksheet>')


GOOD = ['写实。', '<场景1>：室内。', '<主体1>：一个人。', '无。', '<主体1>站立。', '0-1秒，<主体1>站立在<场景1>中。']


class SplitInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'input.xlsx'

    def tearDown(self):
        self.temp.cleanup()

    def test_all_sheets_exact_fields_and_literal_ids_without_report_leak(self):
        rows=[list(HEADERS)+['问题明细','备注'], ['001',*GOOD,'REPORT_SECRET','NOTE_SECRET'], [None]*7+['REPORT_ONLY']]
        workbook(self.source,[('数据堆0',rows),('数据堆1',[list(HEADERS),['002',*GOOD]])])
        original=self.source.read_bytes()
        result=prepare_split_input(self.source)
        records=[json.loads(line) for line in result['path'].read_text().splitlines()]
        self.assertEqual([r['id'] for r in records],['001','002'])
        self.assertEqual([(r['source_sheet'],r['source_row']) for r in records],[('数据堆0',2),('数据堆1',2)])
        self.assertNotIn('SECRET',result['path'].read_text())
        self.assertNotIn('REPORT_ONLY',result['path'].read_text())
        self.assertEqual({k:v.strip() for k,v in qa.parse_caption_sections(records[0]['caption']).items()},dict(zip(FIELDS,GOOD)))
        self.assertEqual(self.source.read_bytes(),original)
        before=result['path'].stat().st_mtime_ns
        prepare_split_input(self.source)
        self.assertEqual(result['path'].stat().st_mtime_ns,before)
        selected=prepare_split_input(self.source,sheet_name='数据堆1')
        self.assertEqual(selected['records'],1)

    def test_other_layout_not_guessed_and_duplicate_headers_rejected(self):
        workbook(self.source,[('表',[['备注','自然指令v2'],['x','普通内容']])])
        self.assertIsNone(prepare_split_input(self.source))
        workbook(self.source,[('表',[list(HEADERS)+['主体v2'],['1',*GOOD,'extra']])])
        with self.assertRaisesRegex(ValueError,'标题重复'):
            prepare_split_input(self.source)

    def test_formulas_invalid_types_and_embedded_headers_fail_before_network(self):
        for value in [('A1',),42,'人。\n## 主体\n其他内容。']:
            with self.subTest(value=value):
                fields=list(GOOD); fields[0]=value
                workbook(self.source,[('表',[list(HEADERS),['1',*fields]])])
                with self.assertRaisesRegex(ValueError,'未调用 API'):
                    prepare_split_input(self.source)

    def test_missing_id_stays_missing_and_explicit_wrong_id_fails(self):
        workbook(self.source,[('表',[list(HEADERS),[None,*GOOD]])])
        result=prepare_split_input(self.source)
        self.assertIsNone(json.loads(result['path'].read_text())['id'])
        with self.assertRaisesRegex(ValueError,'ID 字段'):
            prepare_split_input(self.source,id_column='问题明细')

    def test_plain_same_field_prefix_keeps_actual_content(self):
        fields=list(GOOD); fields[1]='场景：纯白背景。'
        workbook(self.source,[('表',[list(HEADERS),['001',*fields]])])
        original=self.source.read_bytes()
        result=prepare_split_input(self.source)
        caption=json.loads(result['path'].read_text())['caption']
        self.assertEqual(qa.parse_caption_sections(caption)['场景'].strip(),'纯白背景。')
        self.assertIn('场景：纯白背景。',caption)
        self.assertEqual(self.source.read_bytes(),original)

    def test_web_full_mode_uses_model_for_all_sheets_and_restores_history(self):
        workbook(self.source,[(f'数据堆{i}',[list(HEADERS)+['问题明细'],[f'00{i}',*GOOD,'REPORT_SECRET']]) for i in range(2)])
        client=FakeAsyncClient(semantic_reply(),semantic_reply())
        agent_class=qa.CaptionQAAgent
        def factory(**kwargs):
            self.assertFalse(kwargs['dry_run'])
            return agent_class(async_client=client,**kwargs)
        app=web.App(self.root/'out',self.root/'work')
        try:
            payload=self.source.read_bytes()
            with patch.object(web,'read_saved_key',return_value='tp-test'), patch.object(qa,'CaptionQAAgent',side_effect=factory), patch.object(web,'export_repair_table',return_value={}):
                item=app.create_job('分列.xlsx',io.BytesIO(payload),len(payload),review_first=False, dry_run=False)
                app.executor.submit(lambda:None).result(timeout=10)
            snap=app.jobs[item['id']].snapshot()
            self.assertEqual(snap['status'],'done',snap['error'])
            self.assertEqual(snap['total'],2)
            self.assertEqual(snap['check_report']['semantic_checked'],2)
            self.assertEqual(len(client.calls),2)
            for call in client.calls:
                self.assertEqual(set(request_sections(call)),set(FIELDS))
                self.assertEqual(json.loads(call['messages'][1]['content'])['read_only_sections'],['画面元素'])
                self.assertNotIn('REPORT_SECRET',json.dumps(call,ensure_ascii=False))
            self.assertEqual(app.jobs[item['id']].source.read_bytes(),payload)
        finally:
            app.close()
        restored=web.App(self.root/'out',self.root/'work')
        try:
            self.assertEqual(restored.jobs[item['id']].snapshot()['status'],'done')
        finally:
            restored.close()

    def test_ui_default_explicitly_calls_api(self):
        html=(Path(web.ROOT)/'web/index.html').read_text()
        self.assertIn('value="full" checked',html)
        self.assertNotIn('value="static" checked',html)
        self.assertIn('调用 MiMo API',html)


if __name__=='__main__':
    unittest.main()
