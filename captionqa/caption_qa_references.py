"""Validate Caption reference indices against source media, without network I/O."""
import json
import re

MEDIA = ('图片', '视频', '音频')
NAMES = {
    '图片': {'图片', '参考图片', 'ref图片', 'refimages', 'refimage', 'referenceimages', 'referenceimage', 'referenceimageurls'},
    '视频': {'视频', '参考视频', 'ref视频', 'refvideos', 'refvideo', 'referencevideos', 'referencevideo', 'referencevideourls'},
    '音频': {'音频', '参考音频', 'ref音频', 'refaudios', 'refaudio', 'referenceaudios', 'referenceaudio', 'referenceaudiourls'},
}
EXTENSION = re.compile(r'\.(?:jpe?g|png|webp|gif|bmp|avif|heic|mp4|mov|webm|mkv|avi|mp3|wav|m4a|aac|ogg|flac)(?:[?#].*)?$', re.I)
URL = re.compile(r'https?://[^\s<>"\[\]{}，；,;]+', re.I)


def _key(text):
    return re.sub(r'[\s_\-（）()]', '', str(text)).lower()


def detect_columns(records):
    """Recognize explicit media columns, including the app's two-row headers."""
    result = {'media': {}, 'counts': None, 'order': None}
    for record in records:
        for key in record:
            compact = _key(key)
            for media, names in NAMES.items():
                if compact in names: result['media'].setdefault(media, []).append(key)
            if compact in {'素材数', 'refcounts', 'referencecounts', '参考素材数量'}:
                result['counts'] = key
        # Restrict secondary-header inference to an actual grouped ref header.
        if '参考素材' in record and sum(str(v).strip() in MEDIA for v in record.values()) >= 2:
            group = False
            for key, value in record.items():
                if key == '参考素材': group = True
                elif group and not key.startswith('__column_'): group = False
                if group and str(value).strip() in MEDIA:
                    result['media'].setdefault(str(value).strip(), []).append(key)
            count_name = result['counts']
            if count_name:
                order = re.split(r'[/／、,，|｜]', str(record.get(count_name, '')))
                aliases = {'图':'图片','视':'视频','音':'音频', **{m:m for m in MEDIA}}
                order = [aliases.get(v.strip()) for v in order]
                if len(order)==3 and set(order)==set(MEDIA): result['order'] = order
    result['media'] = {kind:list(dict.fromkeys(keys)) for kind,keys in result['media'].items()}
    return result


def count_cell(value):
    """Unknown nonempty formats stay unknown, never silently counted as zero."""
    if value is None or value == '': return 0
    if isinstance(value, list):
        counts = [count_cell(item) for item in value]
        return sum(counts) if all(n is not None for n in counts) else None
    if isinstance(value, dict):
        for key in ('attachments', 'files', 'items', 'urls'):
            if key in value: return count_cell(value[key])
        for key in ('url', 'path', 'filename', 'file_name', 'name'):
            if isinstance(value.get(key), str) and (URL.fullmatch(value[key]) or EXTENSION.search(value[key])): return 1
        return None
    if not isinstance(value, str): return None
    value = value.strip()
    if not value or value.lower() in {'[]', 'null', 'none', '无', '-', '—'}: return 0
    if value.startswith(('[', '{')):
        try: return count_cell(json.loads(value))
        except (ValueError, TypeError): return None
    links = list(URL.finditer(value))
    if links:
        remainder = URL.sub('', value)
        # Allow common attachment-list wrappers and ordinal labels only.
        if re.sub(r'[\s,，;；|｜\[\]()（）\d.、:-]', '', remainder): return None
        return len(links)
    parts = [v.strip() for v in re.split(r'[\n\r,，;；|｜]', value) if v.strip()]
    if parts and all(EXTENSION.search(v) for v in parts): return len(parts)
    return None


def inventory(record, columns):
    counts = {}; notes = []; declared = {}
    if columns.get('counts') and columns.get('order'):
        raw = str(record.get(columns['counts'], ''))
        if re.fullmatch(r'\s*\d+\s*[/／]\s*\d+\s*[/／]\s*\d+\s*', raw):
            declared = dict(zip(columns['order'], map(int, re.findall(r'\d+',raw))))
    for media in MEDIA:
        keys = columns.get('media', {}).get(media, [])
        parts = [count_cell(record.get(key)) for key in keys]
        if keys and all(n is not None for n in parts):
            counts[media] = sum(parts)
            if media in declared and declared[media] != counts[media]:
                notes.append(f'{media}素材数量不一致：素材数列为 {declared[media]}，实际参考素材列为 {counts[media]}；按实际参考素材列校验。')
        elif keys:
            counts[media] = None
            notes.append(f'{media}参考素材格式无法确定数量，请核对源素材列。')
        else: counts[media] = declared.get(media)
    return counts, notes


def findings(sections, record, columns):
    from captionqa.caption_qa_symbols import mask_literals
    counts, notes = inventory(record, columns)
    issues = []
    for field, text in sections.items():
        view = mask_literals(text)
        view = URL.sub(lambda m:' '*len(m[0]),view)
        seen = set()
        # Match both <图片6> and bare 图片6; prevent matching URL/file names.
        for match in re.finditer(r'(?<![A-Za-z0-9_/])(图片|视频|音频)[ \t]*([0-9]+)(?![0-9A-Za-z_])', view):
            media, number = match[1], int(match[2]); count = counts.get(media)
            if count is None or 1 <= number <= count or (media,number) in seen: continue
            seen.add((media, number))
            allowed = f'{media}1～{media}{count}' if count else f'没有可引用的{media}'
            reason = f'本行实际有 {count} 个 ref {media}素材，却引用了 {media}{number}；{allowed}。请核对源素材和引用编号，不自动猜改。'
            issues.append({'source':'static','section':field,'original_snippet':text[match.start():match.end()],
                           'issue_type':'reference_out_of_range','suggestion':reason,'can_auto_fix':False,'resolved':False})
    return issues, counts, notes


def attach_report(report, record, columns):
    if report.get('qa_scope') == 'input_error' or report.get('status') == 'SKIPPED_NOT_HUMAN_CAPTION': return report
    from captionqa.caption_qa_agent import parse_caption_sections
    try: sections = parse_caption_sections(report['revised_caption'])
    except ValueError: return report
    issues, counts, notes = findings(sections, record, columns)
    report['reference_validation'] = {'counts':counts, 'notes':notes,
        'status':'needs_confirmation' if issues or notes else 'checked' if all(v is not None for v in counts.values()) else 'unavailable'}
    report['issues'].extend(issues); report['static_errors'].extend(issues); report['remaining_static_errors'].extend(issues)
    if notes:
        # Surface uncertain inventory as a review item, not a false PASS.
        field = '输入描述' if '输入描述' in sections else next(iter(sections))
        for note in notes:
            item = {'source':'static','section':field,'original_snippet':'','issue_type':'reference_inventory',
                    'suggestion':note,'can_auto_fix':False,'resolved':False}
            report['issues'].append(item); report['static_errors'].append(item); report['remaining_static_errors'].append(item)
    if issues or notes:
        report['need_human_review'] = True; report['status'] = 'NEED_HUMAN_REVIEW'
        report['human_review_reason'] = '；'.join(filter(None, [report.get('human_review_reason'), *(i['suggestion'] for i in issues), *notes]))
    return report
