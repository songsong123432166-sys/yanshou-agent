"""One resolver for approved operations and manual drafts; no file writes."""
import captionqa.caption_qa_agent as qa
from captionqa.caption_qa_review_audio import touches_audio
from captionqa.caption_qa_review_model import AUTO_WORKFLOW, FieldResolution


def resolve_field(document, field, issues):
    field_id = field["field_id"]
    old = field["original"]
    if field_id in document["field_edits"]:
        new = document["field_edits"][field_id]
        new = qa._pad_like(old, new)
        mode = "human"
    else:
        selected = [i for i in issues if i['decision'] in {'approve', 'auto'}]
        from captionqa.caption_qa_candidate_context import normalized_candidate
        selected = [normalized_candidate(i,old) for i in selected]
        from captionqa.caption_qa_candidates import valid_candidate, candidate_operations, PUNCTUATION_POLICY, EXPLICIT_PUNCTUATION_POLICY, REWRITE_POLICY
        from captionqa.caption_qa_suggestions import POLICY as SUGGESTION_POLICY
        for issue in selected:
            from captionqa.caption_qa_labels import POLICY as SEPARATOR_POLICY
            if issue.get('separator_policy')==SEPARATOR_POLICY:
                from captionqa.caption_qa_labels import separator_operations
                if issue['operations']!=separator_operations(field['name'],old):
                    raise ValueError('标签分隔符修改超出明确格式规范，已停止回填。')
            if issue.get('requires_confirmation'):
                expected=candidate_operations(issue,old)
                if len(issue['operations'])!=len(expected) or any(
                    not any(all(candidate.get(key)==operation.get(key) for key in candidate) for operation in issue['operations'])
                    for candidate in expected):
                    raise ValueError('候选纠错必须完整执行已展示的指定改法，已停止回填。')
            for operation in issue['operations']:
                if operation.get('review_only') or issue.get('requires_confirmation'):
                    if issue['decision']!='approve' or not valid_candidate(issue,old,operation):
                        raise ValueError('候选纠错必须经人工确认且与展示的指定改法一致，已停止回填。')
        operations = [op for issue in selected for op in issue["operations"]]
        operations = sorted({(o['start'],o['end'],o['old'],o['new']):o for o in operations}.values(),key=lambda o:(o['start'],o['end']))
        for left, right in zip(operations, operations[1:]):
            if right["start"] < left["end"] or right["start"] == left["start"]:
                raise ValueError("同一字段的已确认修改存在冲突，请人工统一定稿。")
        from captionqa.caption_qa_candidate_context import contextual_operations
        operations = contextual_operations(old,operations)
        new = old
        from captionqa.caption_qa_width import normalize, valid_operation
        audio_text=normalize(field['name'],old,exempt=field['name'] in qa.PUNCTUATION_EXEMPT_SECTIONS) if document['workflow_version']==AUTO_WORKFLOW else old
        for op in reversed(operations):
            candidate_width=(op.get('candidate_policy')==PUNCTUATION_POLICY and
                             any(i['decision']=='approve' and i.get('requires_confirmation') and valid_candidate(i,old,op) for i in selected))
            width_ok=(bool(op.get('width_rule')) and (document['workflow_version']==AUTO_WORKFLOW or candidate_width)
                      and valid_operation(field['name'],old,op,exempt=field['name'] in qa.PUNCTUATION_EXEMPT_SECTIONS))
            width_ok=width_ok or (candidate_width and bool(op.get('confirmed_audio_width')))
            confirmed_candidate=(op.get('candidate_policy') in {EXPLICIT_PUNCTUATION_POLICY,REWRITE_POLICY,SUGGESTION_POLICY} and
                                 any(i['decision']=='approve' and i.get('requires_confirmation') and valid_candidate(i,old,op) for i in selected))
            if old[op["start"]:op["end"]] != op["old"] or (op.get('width_rule') and not width_ok) or (touches_audio(audio_text, op["start"], op["end"]) and not (width_ok or confirmed_candidate)):
                raise ValueError("指定修改与原文或音频保护范围不一致，已停止回填。")
            new = new[:op["start"]] + op["new"] + new[op["end"]:]
        mode = ('approved_candidate' if any(i.get('requires_confirmation') for i in selected) else
                'automatic_rule' if selected and all(i['decision'] == 'auto' for i in selected) else 'approved_rule')

    editor = document['field_edits'].get(field_id, new.strip())
    return FieldResolution(field_id, field['row_index'], field['sample_id'], field['name'],
                           old, new, editor, mode)


def resolve_fields(document):
    """Capture per-field blockers so a bad draft can still be opened and fixed."""
    groups = {}
    for issue in document['issues']:
        groups.setdefault(issue['field_id'], []).append(issue)
    results = []
    for field in document['fields']:
        try:
            result = resolve_field(document, field, groups.get(field['field_id'], []))
        except ValueError as exc:
            original = field['original']
            result = FieldResolution(field['field_id'], field['row_index'], field['sample_id'],
                field['name'], original, original,
                document['field_edits'].get(field['field_id'], original.strip()), 'blocked', str(exc))
        results.append(result)
    return tuple(results)


def changes_from_fields(document, fields):
    changes = {}
    originals = {f['field_id']:f for f in document['fields']}
    for result in fields:
        if result.error:
            raise ValueError(result.error)
        if result.changed:
            field = originals[result.field_id]
            changes.setdefault(result.row_index, []).append({**field, 'after':result.final_text, 'mode':result.mode})
    return changes


def approved_changes(document):
    """Compatibility API; every consumer shares the same field resolver."""
    return changes_from_fields(document, resolve_fields(document))
