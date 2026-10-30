"""Read-only checks of saved final text; warnings never rewrite a human draft."""
import re
import difflib

from captionqa.caption_qa_review_audio import touches_audio
from captionqa.caption_qa_width import SCREEN


def repeated_punctuation(text):
    return [m for m in re.finditer(r'([。，、；：])\1+', text)
            if not touches_audio(text, m.start(), m.end())
            and not any(s.start() <= m.start() < s.end() for s in SCREEN.finditer(text))]


def assess_review(document, fields, rows):
    """Recheck original static findings against the whole final Caption.

    An unchanged semantic snippet is only a request to verify, never a claim
    that a model's diagnosis was right. Explicit keep/reject decisions prevail.
    """
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_review_state import human_review_issues
    resolved = {f.field_id:f for f in fields}
    final_rows = {r.row_index:r for r in rows}
    checks = {}
    static = {}
    for issue in human_review_issues(document):
        if issue['decision'] not in {'manual', 'approve'}:
            continue
        field = resolved[issue['field_id']]
        row = final_rows[field.row_index]
        if field.error or row.errors:
            continue
        # Historical parser versions sometimes included the following old
        # header/body in a field. A stale span cannot support a new diagnosis.
        original_sections=qa.parse_caption_sections(row.original)
        if original_sections.get(field.name) != field.original:
            continue
        message = ''
        if issue.get('origin') == 'static':
            if row.row_index not in static:
                static[row.row_index] = qa.run_static_rules(qa.parse_caption_sections(row.final_text))
            for finding in static[row.row_index]:
                if finding['issue_type'] != issue['kind']:
                    continue
                cross = issue['kind'] in {'ghost_subjects', 'missing_subjects'}
                if (cross or finding['section'] == field.name) and finding['original_snippet'] == issue['snippet']:
                    message = '本地复检仍发现原问题，请核对最终 Caption；其他字段中的引用也会一并检查。'
                    break
        elif issue['decision'] == 'manual':
            snippet = issue.get('snippet', '').strip()
            # Do not flag insertion fixes whose unchanged snippet is merely an
            # anchor. For semantic text, compare letters after punctuation is
            # removed, so a comma added at a snippet boundary is not missed.
            language = issue['kind'] in {'无歧义错别字', '语法病句与语句不通顺', '隐式漏挂主体标签'}
            if field.final_text.strip() == field.original.strip():
                message = '人工稿与原字段相同，文字未改变。请确认此疑点是否应保留或判为误报。'
            elif language and snippet and snippet in field.final_text:
                # A new tag/changed definition elsewhere can resolve a
                # reference issue without changing the cited words.
                # An inserted comma immediately beside the cited snippet can
                # be the entire fix. Do not treat that unchanged anchor as a
                # missed edit merely because punctuation lies outside it.
                ranges=[m.span() for m in re.finditer(re.escape(snippet),field.original)]
                boundary_symbols=r'[<>]+' if issue['kind']=='隐式漏挂主体标签' else r'[\s。，、；：！？<>]+'
                boundary_fix=any(code!='equal' and a==b and any(a in (p,q) for p,q in ranges)
                    and re.fullmatch(boundary_symbols,field.final_text[c:d])
                    for code,a,b,c,d in difflib.SequenceMatcher(None,field.original,field.final_text,autojunk=False).get_opcodes())
                definition_changed=any(f.row_index==field.row_index and f.name=='主体' and f.changed for f in fields)
                if not boundary_fix and not (issue['kind']=='隐式漏挂主体标签' and definition_changed):
                    message = '人工稿已保存，但原疑点片段仍在。请核对是否已通过上下文解决；此提示不认定 AI 判断正确。'
        if message:
            checks[issue['issue_id']] = {'issue_id':issue['issue_id'], 'field_id':field.field_id,
                'sample_id':field.sample_id, 'field':field.name, 'message':message}
    warnings = list(checks.values())
    for field in fields:
        if field.error or field.mode != 'human':
            continue
        for match in repeated_punctuation(field.final_text):
            warnings.append({'field_id':field.field_id, 'sample_id':field.sample_id, 'field':field.name,
                'message':f'人工稿中存在连续标点「{match[0]}」，请核对；程序不会覆盖人工稿。',
                'excerpt':field.final_text[max(0,match.start()-25):match.end()+25]})
    return warnings
