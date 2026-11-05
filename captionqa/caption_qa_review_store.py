"""Storage adapter for existing review JSON/checkpoints, without HTTP concerns."""
from dataclasses import dataclass
import json
from pathlib import Path

import captionqa.caption_qa_agent as qa
from captionqa.caption_qa_review_state import summary
from captionqa.caption_qa_review_plan import compile_review
from captionqa.caption_qa_table_io import iter_table_records


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_binding(document, source, input_source, cache):
    for key, path in (("source_sha256", source), ("input_sha256", input_source), ("cache_sha256", cache)):
        if document[key] != qa._source_sha256(Path(path)):
            raise ValueError("原文或检测版本已经变化，请重新检测并审核；拒绝套用旧审核结果。")


def source_records(document, input_source):
    return {index:record for index,record,error in iter_table_records(
        input_source, sheet_name=document['sheet']) if not error}


def read_review_summary(cache):
    """The saved review is authoritative; a stale derived cache is never shown."""
    path = Path(cache).with_suffix('.review.json')
    if not path.is_file():
        return {}
    try:
        return summary(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return {}


@dataclass(frozen=True)
class ReviewStore:
    source: Path
    input_source: Path
    cache: Path
    column: str | None = None
    id_column: str | None = None
    sheet: str | None = None

    @property
    def path(self):
        return self.cache.with_suffix('.review.json')

    def save(self, document):
        atomic_json(self.path, document)
        # This cache is optional. Its failure must not report a successfully
        # persisted review as unsaved; readers derive status from the review.
        try:
            atomic_json(self.cache.with_suffix('.review_summary.json'), summary(document))
        except OSError:
            pass

    def load(self):
        from captionqa.caption_qa_review_resume import resume_review
        resume_review(self)
        from captionqa.caption_qa_review import make_review
        if self.path.exists():
            document = json.loads(self.path.read_text())
            validate_binding(document, self.source, self.input_source, self.cache)
            from captionqa.caption_qa_candidates import enrich_candidates
            from captionqa.caption_qa_candidate_context import normalize_saved_candidates
            contextual = normalize_saved_candidates(document)
            added = enrich_candidates(document)
            from captionqa.caption_qa_labels import enrich_separator_fixes, POLICY as SEPARATOR_POLICY
            separators = enrich_separator_fixes(document)
            from captionqa.caption_qa_time_layout import enrich_time_layout, POLICY as TIME_POLICY
            layout=enrich_time_layout(document,source_records(document,self.input_source))
            from captionqa.caption_qa_labels import enrich_pipe_width, PIPE_POLICY
            pipes=enrich_pipe_width(document,source_records(document,self.input_source))
            from captionqa.caption_qa_contract import enrich as enrich_contract, POLICY as CONTRACT_POLICY
            previous_policy=document.get('format_policy')
            contract=enrich_contract(document,source_records(document,self.input_source))
            if added or separators or contextual or layout or pipes or contract or previous_policy!=CONTRACT_POLICY:
                document['revision'] += 1
                if contract or previous_policy!=CONTRACT_POLICY:
                    document['events'].append({'revision':document['revision'],'decision':'metadata_reference_checks',
                        'policy':CONTRACT_POLICY,'issue_ids':contract,'reason':'中文标签、标题格式及 ref 引用校验；已有人工决定和定稿保留。'})
                if contextual:
                    document['events'].append({'revision':document['revision'],
                        'decision':'candidate_context_normalized', 'issue_ids':contextual,
                        'reason':'避免候选重复插入原文已有标点；人工决定与人工稿保留。'})
                policies = sorted({i['candidate_policy'] for i in document['issues'] if i.get('requires_confirmation')})
                if added:
                    document['events'].append({'revision':document['revision'], 'decision':'candidates_available',
                                              'policies':policies, 'issues':added})
                for issue_id in separators:
                    issue=next(i for i in document['issues'] if i['issue_id']==issue_id)
                    document['events'].append({'revision':document['revision'],'decision':issue['decision'],
                                              'issue_id':issue_id,'policy':SEPARATOR_POLICY})
                for issue_id in layout:
                    issue=next(i for i in document['issues'] if i['issue_id']==issue_id)
                    document['events'].append({'revision':document['revision'],'decision':issue['decision'],
                                              'issue_id':issue_id,'policy':TIME_POLICY})
                for issue_id in pipes:
                    issue=next(i for i in document['issues'] if i['issue_id']==issue_id)
                    document['events'].append({'revision':document['revision'],'decision':issue['decision'],
                                              'issue_id':issue_id,'policy':PIPE_POLICY})
                self.save(document)
        else:
            from captionqa.caption_qa_id_list import detect_id_column
            column = self.column or qa.detect_caption_column(self.input_source, sheet_name=self.sheet)
            id_column = 'id' if self.input_source != self.source else detect_id_column(
                self.input_source, sheet_name=self.sheet, column=self.id_column)
            document = make_review(self.cache, self.source, self.input_source,
                                   caption_column=column, id_column=id_column, sheet=self.sheet)
            self.save(document)
        return document

    def plan(self, document):
        validate_binding(document, self.source, self.input_source, self.cache)
        return compile_review(document, source_records(document, self.input_source))

    def payload(self, document):
        return {'review':document, 'summary':summary(document), 'projection':self.plan(document).public()}
