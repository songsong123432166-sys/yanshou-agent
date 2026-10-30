"""Compile one source-bound final view for review UI and table exporters.

This module reads no files and mutates no review state. Unapproved findings stay
as text; an invalid approved draft becomes an explicit blocker, never a partial
export. File-format writers consume only a validated plan.
"""
from dataclasses import dataclass, replace
import hashlib
import json

import captionqa.caption_qa_agent as qa
from captionqa.caption_qa_review_edits import resolve_fields, changes_from_fields
from captionqa.caption_qa_review_model import CaptionResolution, DECISION_LABELS, PROJECTION_VERSION, freeze, thaw
from captionqa.caption_qa_review_state import highlighted_rows, automatic_issue_ids, human_review_issues


def document_fingerprint(document):
    # Export caches bind both the saved review and the resolver contract.
    # A code fix must not reuse an older file produced by a previous resolver.
    raw = json.dumps({'resolver_version':'review-export-v2', 'document':document},
                     sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ReviewPlan:
    revision: int
    fingerprint: str
    source_sha256: str
    fields: tuple
    rows: tuple
    _highlights: object
    _changes: object
    _audit: object
    _rechecks: object
    _blockers: object
    _review_checks: object
    automatic_ids: tuple
    visible_ids: tuple

    @property
    def highlights(self):return thaw(self._highlights)

    @property
    def changes(self):return thaw(self._changes)

    @property
    def audit(self):return thaw(self._audit)

    @property
    def rechecks(self):return thaw(self._rechecks)

    @property
    def blockers(self):return thaw(self._blockers)

    @property
    def review_checks(self):return thaw(self._review_checks)

    def require_valid(self):
        if self.blockers:
            item = self.blockers[0]
            raise ValueError(f"题目 {item['sample_id'] or 'ID缺失'} · {item['field']}：{item['message']}")
        return self

    @property
    def captions(self):
        return {row.row_index:row.final_text for row in self.rows if row.changed}

    def public(self):
        # The browser does not receive source records, machine captions or a
        # second implementation of patch application. It receives final fields.
        return {'version':PROJECTION_VERSION, 'review_revision':self.revision,
                'fingerprint':self.fingerprint, 'source_sha256':self.source_sha256,
                'fields':{field.field_id:field.public() for field in self.fields},
                'blockers':list(self.blockers), 'export_ready':not self.blockers,
                'review_checks':list(self.review_checks),
                'review_check_fields':len({w['field_id'] for w in self.review_checks}),
                'changed_captions':len(self.captions),
                'highlighted_captions':len(self.highlights),
                'automatic_issue_ids':list(self.automatic_ids),
                'visible_issue_ids':list(self.visible_ids),
                'decision_labels':dict(DECISION_LABELS)}


def compile_review(document, records):
    """Resolve saved decisions once, then rebuild only changed Caption bodies."""
    resolved = {field.field_id:field for field in resolve_fields(document)}
    grouped = {}
    for field in document['fields']:
        grouped.setdefault(field['row_index'], []).append(field['field_id'])
    highlights = highlighted_rows(document)
    rows, rechecks = [], []
    for index, field_ids in grouped.items():
        fields = [resolved[key] for key in field_ids]
        sample_id = fields[0].sample_id
        original, final, errors = '', '', [f.error for f in fields if f.error]
        try:
            if index not in records:
                raise ValueError('来源行不存在，请重新核对原文件。')
            original = records[index].get(document['caption_column'])
            if not isinstance(original, str):
                raise ValueError('来源Caption不是文本，无法应用审核结果。')
            final = original
            if not errors:
                changed = [field for field in fields if field.changed]
                sections = qa.parse_caption_sections(original)
                for field in changed:
                    if sections.get(field.name) != field.original:
                        raise ValueError('字段原文已变化，拒绝覆盖。')
                updated = dict(sections)
                updated.update({field.name:field.final_text for field in changed})
                if changed:
                    final = qa.rebuild_caption(original, updated)
                    reparsed = qa.parse_caption_sections(final)
                    if list(reparsed) != list(sections) or any(reparsed.get(k) != v for k,v in updated.items()):
                        raise ValueError('改稿改变了字段标题或边界，请仅填写字段正文。')
                if final != original or index in highlights:
                    rechecks.append({'sample_id':sample_id, 'row_index':index,
                        'remaining_static_errors':qa.run_static_rules(qa.parse_caption_sections(final)),
                        'semantic_rechecked':False})
        except ValueError as exc:
            errors.append(str(exc))
            # Associate row reconstruction errors with their changed fields so
            # reviewers can fix the exact draft instead of losing the review UI.
            changed = [f for f in fields if f.changed]
            targets = [f for f in changed if f.field_id in document['field_edits']] or changed or fields
            for field in targets:
                if not field.error:
                    resolved[field.field_id] = replace(field, final_text=field.original,
                                                       mode='blocked', error=str(exc))
            final = original
        rows.append(CaptionResolution(index, sample_id, original, final,
                    tuple(field_ids), index in highlights, tuple(errors)))
    fields = tuple(resolved[field['field_id']] for field in document['fields'])
    blockers = tuple({'field_id':f.field_id, 'row_index':f.row_index,
                      'sample_id':f.sample_id, 'field':f.name, 'message':f.error}
                     for f in fields if f.error)
    changes = changes_from_fields(document, [f for f in fields if not f.error])
    audit = []
    for row_fields in changes.values():
        for field in row_fields:
            audit.append({'题目 ID':field['sample_id'], '字段':field['name'], '修改来源':field['mode'],
                          '修改前':field['original'], '修改后':field['after'], 'field_id':field['field_id'],
                          'source_sheet':field['source_sheet'], 'source_row':field['source_row']})
    from captionqa.caption_qa_review_checks import assess_review
    review_checks=assess_review(document,fields,rows)
    return ReviewPlan(document['revision'], document_fingerprint(document), document['source_sha256'],
        fields, tuple(rows), freeze(highlights), freeze(changes), freeze(audit), freeze(rechecks), freeze(blockers),
        freeze(review_checks),
        tuple(sorted(automatic_issue_ids(document))),
        tuple(issue['issue_id'] for issue in human_review_issues(document)))
