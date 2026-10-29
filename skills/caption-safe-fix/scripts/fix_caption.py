#!/usr/bin/env python3
"""Offline Caption-only correction. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import sys
from uuid import uuid4
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import zipfile

import _rules as rules


CAPTION_HEADERS = {'caption', '人标caption', '人工caption', '人标', '人工标注'}
ID_HEADERS = {'id', '样本id', '样本编号', 'sample_id', 'sampleid', '任务id', 'task_id'}


def normalized(value):
    return re.sub(r'\s+', '', str(value or '')).lower()


def column_number(letters):
    number = 0
    for char in letters.upper():
        number = number * 26 + ord(char) - 64
    return number


def select_column(rows, requested, headers, *, excel=False, required=True):
    candidates = set()
    for _, row in rows[:10]:
        for number, value in row.items():
            if normalized(value) == normalized(requested) if requested else normalized(value) in headers:
                candidates.add(number)
    if requested and len(candidates) == 0:
        if requested.isdigit() and int(requested) > 0:
            return int(requested)
        if excel and re.fullmatch(r'[A-Za-z]{1,3}', requested) and column_number(requested) <= 16384:
            return column_number(requested)
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates and not required:
        return None
    raise ValueError('不能唯一确认列，请明确指定 Caption/ID 列名、列序号或 XLSX 列字母；不会改查其他列。')


class Processor:
    def __init__(self):
        self.counts = Counter(total=0, changed=0, review=0, skipped=0)
        self.pending, self.skipped, self.missing_ids = {}, [], []

    def process(self, value, location, identifier=None, *, single=False, formula=False):
        self.counts['total'] += 1
        if formula or not isinstance(value, str) or not value.strip() or (not single and not rules.is_human_caption(value)):
            reason = 'Caption 是公式，未改写' if formula else '空值、非文本、表头或非支持的结构化 Caption'
            self.counts['skipped'] += 1
            self.skipped.append({'location': location, 'reason': reason})
            return value
        try:
            sections = rules.parse_caption_sections(value)
        except ValueError as exc:
            self.counts['skipped'] += 1
            self.skipped.append({'location': location, 'reason': str(exc)})
            return value
        fixed = rules.apply_static_fixes(sections)
        revised = rules.rebuild_caption(value, fixed)
        # Idempotence is required before a change can leave the process.
        if rules.apply_static_fixes(fixed) != fixed:
            raise ValueError('规则复检不一致，停止输出；原文件未修改。')
        if revised != value:
            self.counts['changed'] += 1
        remaining = rules.run_static_rules(fixed)
        if remaining:
            self.counts['review'] += 1
            notes = list(dict.fromkeys(f"【{item['section']}】{item['suggestion']}" for item in remaining))
            usable = type(identifier) in (str, int) and str(identifier).strip() and not re.search(r'[\r\n]', str(identifier))
            if usable:
                bucket = self.pending.setdefault(str(identifier), [])
                bucket.extend(note for note in notes if note not in bucket)
            else:
                self.missing_ids.append({'location': location, 'errors': notes})
        return revised


def utf8(raw):
    try:
        return raw.decode('utf-8-sig'), raw.startswith(b'\xef\xbb\xbf')
    except UnicodeDecodeError as exc:
        raise ValueError('文件不是 UTF-8 编码，请确认编码后另存副本；未修改原文件。') from exc


def transform_csv(raw, args, processor):
    text, bom = utf8(raw)
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=',\t;')
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text, newline=''), dialect))
    indexed = [(i + 1, {j + 1: v for j, v in enumerate(row)}) for i, row in enumerate(rows)]
    column = select_column(indexed, args.column, CAPTION_HEADERS)
    identifier = select_column(indexed, args.id_column, ID_HEADERS, required=bool(args.id_column))
    if not any(column <= len(row) for row in rows):
        raise ValueError('指定 Caption 列不存在。')
    for index, row in enumerate(rows, 1):
        if column <= len(row):
            sample = row[identifier - 1] if identifier and identifier <= len(row) else None
            row[column - 1] = processor.process(row[column - 1], f'CSV第{index}行第{column}列', sample)
        else:
            processor.process(None, f'CSV第{index}行缺少指定Caption列')
    stream = io.StringIO(newline='')
    newline = '\r\n' if '\r\n' in text else '\n'
    writer = csv.writer(stream, dialect, lineterminator=newline)
    writer.writerows(rows)
    return (b'\xef\xbb\xbf' if bom else b'') + stream.getvalue().encode('utf-8')


def local_name(tag):
    return tag.rsplit('}', 1)[-1]


def string_text(element):
    if element is None:
        return ''
    # Exclude phonetic/ruby annotations from the actual cell value.
    return ''.join(child.text or '' for item in element for child in (
        [item] if local_name(item.tag) == 't' else list(item) if local_name(item.tag) == 'r' else [])
        if local_name(child.tag) == 't')


def transform_xlsx(raw, args, processor):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {element.attrib['Id']: element.attrib['Target'] for element in relationships}
        sheets = []
        for element in workbook.iter():
            if local_name(element.tag) == 'sheet':
                relation = next((v for k, v in element.attrib.items() if k.endswith('}id')), None)
                target = targets.get(relation, '')
                path = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/' + target)
                if path not in archive.namelist():
                    raise ValueError('工作表关系无法解析，未修改文件。')
                sheets.append((element.attrib['name'], path))
        chosen = [item for item in sheets if item[0] == args.sheet] if args.sheet else sheets
        if len(chosen) != 1:
            raise ValueError('请用 --sheet 指定唯一工作表，可选：' + '、'.join(name for name, _ in sheets))
        sheet_name, path = chosen[0]
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [string_text(element) for element in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        xml = archive.read(path).decode('utf-8-sig')
        root = ET.fromstring(xml)
        rows, formulas, addresses = {}, set(), {}
        for element in root.iter():
            if local_name(element.tag) != 'c':
                continue
            address = element.get('r', '')
            match = re.fullmatch(r'([A-Z]+)([1-9]\d*)', address)
            if not match:
                raise ValueError('存在无法确认坐标的单元格，未修改文件。')
            column, number = column_number(match[1]), int(match[2])
            children = {local_name(child.tag): child for child in element}
            if 'f' in children:
                formulas.add(address)
            value = children.get('v')
            value = value.text if value is not None else None
            kind = element.get('t')
            if kind == 's':
                value = shared[int(value)]
            elif kind == 'inlineStr':
                value = string_text(children.get('is'))
            elif kind not in {'str', 's', 'inlineStr'} and value is not None:
                # Numeric cells aren't Caption text. ID values remain their original text representation.
                try:
                    value = int(value) if re.fullmatch(r'-?\d+', value) else float(value)
                except ValueError:
                    pass
            rows.setdefault(number, {})[column] = value
            addresses[(number, column)] = address
        indexed = sorted(rows.items())
        column = select_column(indexed, args.column, CAPTION_HEADERS, excel=True)
        identifier = select_column(indexed, args.id_column, ID_HEADERS, excel=True, required=bool(args.id_column))
        if not any(column in row for _, row in indexed):
            raise ValueError('指定 Caption 列不存在。')
        replacements = {}
        for number, row in indexed:
            if column not in row:
                processor.process(None, f'{sheet_name}!第{number}行缺少指定Caption列')
                continue
            address = addresses[(number, column)]
            old = row[column]
            new = processor.process(old, f'{sheet_name}!{address}', row.get(identifier), formula=address in formulas)
            if new != old:
                replacements[address] = new
        replaced = set()
        # Patch only exact cell elements, preserving all unrelated XML namespaces,
        # styles, formulas, links, other worksheets, drawings and archive members.
        pattern = re.compile(r'<(?P<prefix>(?:[\w.-]+:)?)c\b(?P<attrs>[^>]*?)(?:\s*/>|>.*?</(?P=prefix)c\s*>)', re.S)

        def replace(match):
            attrs = match['attrs']
            reference = re.search(r'(?:^|\s)r\s*=\s*([\'"])(.*?)\1', attrs)
            address = reference[2] if reference else None
            if address not in replacements:
                return match[0]
            if address in replaced:
                raise ValueError('同一目标单元格出现重复坐标，未修改文件。')
            replaced.add(address)
            attrs = re.sub(r'\s+t\s*=\s*([\'"]).*?\1', '', attrs)
            prefix = match['prefix']
            body = escape(replacements[address]).replace('\r', '&#13;')
            return f'<{prefix}c{attrs} t="inlineStr"><{prefix}is><{prefix}t xml:space="preserve">{body}</{prefix}t></{prefix}is></{prefix}c>'

        modified = pattern.sub(replace, xml)
        if replaced != set(replacements):
            raise ValueError('无法安全定位所有目标单元格，未修改文件。')
        ET.fromstring(modified)  # Validate XML before any output is written.
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as target:
            for member in archive.infolist():
                content = modified.encode('utf-8') if member.filename == path and replacements else archive.read(member.filename)
                target.writestr(member, content)
        return output.getvalue()


def transform_json(raw, args, processor, *, lines=False):
    text, bom = utf8(raw)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('JSON存在重复字段，不能安全保留原数据；未修改文件。')
            result[key] = value
        return result
    values = [json.loads(line, object_pairs_hook=unique_keys) for line in text.splitlines() if line.strip()] if lines else json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(values, list):
        raise ValueError('JSON 输入必须是记录数组，JSONL 每行必须是对象或 Caption 字符串。')
    for index, record in enumerate(values, 1):
        if isinstance(record, str):
            values[index - 1] = processor.process(record, f'记录{index}')
        elif isinstance(record, dict):
            column = args.column or 'caption'
            if column not in record:
                processor.process(None, f'记录{index}缺少指定字段{column}')
                continue
            identifier = record.get(args.id_column or 'id')
            record[column] = processor.process(record[column], f'记录{index}.{column}', identifier)
        else:
            processor.process(None, f'记录{index}不是对象或文本')
    result = ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values) if lines else json.dumps(values, ensure_ascii=False, indent=2) + '\n'
    return (b'\xef\xbb\xbf' if bom else b'') + result.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--column')
    parser.add_argument('--sheet')
    parser.add_argument('--id-column')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        source = args.input.resolve(strict=True)
        raw = source.read_bytes()
        processor = Processor()
        suffix = source.suffix.lower()
        if suffix == '.xlsx':
            result = transform_xlsx(raw, args, processor)
        elif suffix == '.csv':
            result = transform_csv(raw, args, processor)
        elif suffix in {'.json', '.jsonl'}:
            result = transform_json(raw, args, processor, lines=suffix == '.jsonl')
        elif suffix in {'.txt', '.md'}:
            text, bom = utf8(raw)
            result = (b'\xef\xbb\xbf' if bom else b'') + processor.process(text, '单条Caption', single=True).encode('utf-8')
        else:
            raise ValueError('只支持 XLSX、CSV、JSON、JSONL、TXT、MD；未修改原文件。')
        folder = (args.output_dir or source.parent).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        stem = source.stem[:55] + '_已修正_' + uuid4().hex[:12]
        output = folder / (stem + suffix)
        report = folder / (stem + '_处理说明.json')
        review = folder / (stem + '_待复核.csv')
        data = {'rule_version': rules.AUTO_FIX_POLICY_VERSION,
                'input_sha256': hashlib.sha256(raw).hexdigest(), 'output_sha256': hashlib.sha256(result).hexdigest(),
                'output': str(output), 'review_table': str(review), 'counts': dict(processor.counts),
                'needs_review': [{'id': key, 'errors': notes} for key, notes in processor.pending.items()],
                'missing_review_ids': processor.missing_ids, 'skipped': processor.skipped,
                'scope': '只修正指定 Caption 的既有编号标签格式及段尾标点；语义未验收。'}
        csv_text = io.StringIO(newline='')
        writer = csv.writer(csv_text)
        writer.writerow(['题目 ID', '错误项'])
        for identifier, notes in processor.pending.items():
            # Force literal text when spreadsheet software opens CSV.
            safe_id = "'" + identifier if identifier.startswith(('=', '+', '-', '@')) else identifier
            writer.writerow([safe_id, '\n'.join(f'{i}. {note}' for i, note in enumerate(notes, 1))])
        if source.read_bytes() != raw:
            raise ValueError('处理中输入文件发生变化，停止输出，请重新读取。')
        with output.open('xb') as stream:
            stream.write(result)
        with report.open('xb') as stream:
            stream.write((json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        with review.open('xb') as stream:
            stream.write(b'\xef\xbb\xbf' + csv_text.getvalue().encode('utf-8'))
        print(json.dumps({'output': str(output), 'report': str(report), 'review': str(review), 'counts': dict(processor.counts)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile, ET.ParseError, csv.Error) as exc:
        print('未完成：' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
