"""Identification, human decisions and approved edits are separate operations."""
from __future__ import annotations

from collections import Counter
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re

import caption_qa_agent as qa
from caption_qa_summary import DISABLED

from caption_qa_review_model import (WORKFLOW, PREVIOUS_AUTO_WORKFLOW, AUTO_WORKFLOW,
                                     AUTO_WORKFLOWS, DECISIONS)
from caption_qa_review_audio import audio_spans, touches_audio
from caption_qa_review_state import (automatic_issue_ids, human_review_issues,
                                     highlighted_rows, summary, decide)
from caption_qa_review_edits import approved_changes
from caption_qa_review_store import atomic_json, validate_binding

LABELS = {"missing_punctuation": "缺少段尾标点", "bare_tags": "编号标签格式",
          "missing_subjects": "自然指令漏写已定义主体", "ghost_subjects": "引用未定义主体",
          "duplicate_defined_subjects": "主体重复定义", "skipped_numbering": "主体编号不连续",
          "symbol_format": "内容符号不成对", "label_separator": "多标签拼接符不规范",
          "无歧义错别字": "疑似错别字", "标点符号硬伤": "疑似标点问题",
          "语法病句与语句不通顺": "疑似病句", "隐式漏挂主体标签": "疑似漏挂主体标签",
          "内容符号不规范": "疑似内容符号用错", "裸标签漏写 <>": "疑似编号标签格式",
          "symbol_width": "全角/半角符号"}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


class IdentificationAgent(qa.CaptionQAAgent):
    """Reuse validated suggestions; identification always returns original text."""
    workflow_version = WORKFLOW

    def _identify(self, caption, report):
        proposal = report["revised_caption"]
        report = copy.deepcopy(report)
        report["proposed_caption"] = proposal
        report["revised_caption"] = caption
        report["workflow_version"] = WORKFLOW
        for issue in report["issues"]:
            issue["resolved"] = False
        report["remaining_static_errors"] = qa.run_static_rules(qa.parse_caption_sections(caption)) if report["qa_scope"] != "input_error" else report["remaining_static_errors"]
        if report["issues"] or report["need_human_review"]:
            report["status"], report["need_human_review"] = "NEED_HUMAN_REVIEW", True
        return report

    def check(self, caption):
        return self._identify(caption, super().check(caption))

    async def acheck_single_caption(self, caption):
        return self._identify(caption, await super().acheck_single_caption(caption))


class SafeAutoFixAgent(IdentificationAgent):
    """Execute the existing safe whitelist; keep other findings for humans."""
    workflow_version = AUTO_WORKFLOW

    def __init__(self, *, normalize_width=True, **options):
        super().__init__(**options)
        self.normalize_width=normalize_width
        self.workflow_version=AUTO_WORKFLOW if normalize_width else PREVIOUS_AUTO_WORKFLOW

    def _prepare_width(self, caption):
        if not self.normalize_width:
            return caption,[]
        from caption_qa_width import operations, normalize, unsafe_regions
        try:
            sections=qa.parse_caption_sections(caption)
        except ValueError:
            return caption,[]
        issues=[];updated=dict(sections)
        for name,value in sections.items():
            edits=operations(name,value,exempt=name in qa.PUNCTUATION_EXEMPT_SECTIONS)
            for op in edits:
                item=qa._issue('static',name,value[op['start']:op['end']],'symbol_width',
                               f"{op['width_rule']}：{op['old']} → {op['new']}",True)
                item['width_operations']=[op]
                issues.append(item)
            if name not in qa.PUNCTUATION_EXEMPT_SECTIONS:
                for a,b in unsafe_regions(name,value):
                    issues.append(qa._issue('static',name,value[a:b],'symbol_width',
                                            '包裹符不完整，不能仅凭全角/半角替换补齐；请人工核对。',False))
            updated[name]=normalize(name,value,exempt=name in qa.PUNCTUATION_EXEMPT_SECTIONS)
        return qa.rebuild_caption(caption,updated),issues

    def _with_width_issues(self, report, issues):
        if report['qa_scope']!='input_error':
            report['issues']=issues+report['issues']
            report['static_errors']=issues+report['static_errors']
        return report

    def check(self, caption):
        prepared,issues=self._prepare_width(caption)
        report=qa.CaptionQAAgent.check(self,prepared)
        return self._identify(caption,self._with_width_issues(report,issues))

    async def acheck_single_caption(self, caption):
        prepared,issues=self._prepare_width(caption)
        report=await qa.CaptionQAAgent.acheck_single_caption(self,prepared)
        return self._identify(caption,self._with_width_issues(report,issues))

    def _identify(self, caption, report):
        source_need_human = report['need_human_review']
        report = super()._identify(caption, report)
        report['workflow_version'] = self.workflow_version
        if report['qa_scope'] == 'input_error':
            return report
        document = _review_records({0: {'caption': caption, 'id': '__single__', 'qa_report': report}},
                                   caption_column='caption', id_column='id', source_sha256=digest(caption))
        changes = approved_changes(document)
        sections = qa.parse_caption_sections(caption)
        sections.update({f['name']: f['after'] for f in changes.get(0, [])})
        revised = qa.rebuild_caption(caption, sections)
        if qa.parse_caption_sections(revised) != sections:
            return report
        report['revised_caption'] = revised
        resolved = {i['source_issue_index'] for i in document['issues'] if i['decision'] == 'auto'}
        for index, issue in enumerate(report['issues']):
            issue['resolved'] = index in resolved
        remaining = qa.run_static_rules(sections)
        unresolved = any(not i['resolved'] for i in report['issues'])
        # The inherited flag can include model failures or explanations without
        # an issue. Only clear it when the review builder captured every concern.
        review_needed = source_need_human or unresolved or bool(remaining) or bool(summary(document)['pending'])
        report['remaining_static_errors'] = remaining
        report['need_human_review'] = review_needed
        report['status'] = 'NEED_HUMAN_REVIEW' if review_needed else 'AUTO_FIXED' if revised != caption else 'PASS'
        if not review_needed:
            report['human_review_reason'] = ''
        report['automatic_issue_count'] = len(resolved)
        return report


def sentence_context(text, start, end):
    pairs = {"<": ">", "{": "}", "【": "】", "（": "）", "(": ")", "“": "”", "‘": "’"}
    stack, boundaries, left = [], [], 0
    for index, char in enumerate(text):
        if stack and stack[-1] in "{【":
            if char == pairs[stack[-1]]:
                stack.pop()
            continue
        if char in pairs:
            stack.append(char)
        elif stack and char == pairs[stack[-1]]:
            stack.pop()
        if not stack and char in "。！？!?\n":
            boundaries.append((left, index + 1))
            left = index + 1
    boundaries.append((left, len(text)))
    return list(dict.fromkeys(text[a:b].strip() for a, b in boundaries
                             if a < max(start + 1, end) and b > start and text[a:b].strip()))


def map_position(old, new, position, right=False):
    for op, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if a <= position <= b:
            return c + position - a if op == "equal" else d if right else c
    return len(new)


def _locations(original, cleaned, snippet):
    found = [(m.start(), m.end()) for m in re.finditer(re.escape(snippet), original)] if snippet else []
    if not found and snippet:
        found = [(map_position(cleaned, original, m.start()), map_position(cleaned, original, m.end(), True))
                 for m in re.finditer(re.escape(snippet), cleaned)]
    return found


def _eligible(old, new, kind, item):
    if kind == "bare_tags":
        return (bool(re.search(r"(?:主体|场景|图片|视频|音频)\s*\d+", item["snippet"]))
                and qa._normalize_tags(item["snippet"]) != item["snippet"]
                and all(char in "<> \t" for char in old + new))
    if kind in {"missing_punctuation", "标点符号硬伤", "标点与标签格式"}:
        return qa.EDITABLE_PUNCTUATION_RE.sub("", old) == qa.EDITABLE_PUNCTUATION_RE.sub("", new)
    return kind in {"无歧义错别字", "错别字与衍字"} and bool(item.get("can_auto_fix"))


def make_review(cache, source, input_source, *, caption_column, id_column, sheet=None):
    cache, source, input_source = Path(cache), Path(source), Path(input_source)
    latest = {}
    with cache.open("rb") as stream:
        for line in stream:
            if line.strip() and line.endswith(b"\n"):
                row = json.loads(line)
                latest[row["row_index"]] = row
    document = _review_records(latest, caption_column=caption_column, id_column=id_column,
                               source_sha256=qa._source_sha256(source))
    document.update({
                "input_sha256": qa._source_sha256(input_source), "cache_sha256": qa._source_sha256(cache),
                "sheet": sheet})
    return document


def _review_records(latest, *, caption_column, id_column, source_sha256):
    policies={row.get('qa_report',{}).get('workflow_version') for row in latest.values()}
    automatic=bool(policies & AUTO_WORKFLOWS)
    policy=AUTO_WORKFLOW if AUTO_WORKFLOW in policies else PREVIOUS_AUTO_WORKFLOW if automatic else WORKFLOW
    document = {"workflow_version": policy, "revision": 0,
                "source_sha256": source_sha256, "caption_column": caption_column, "id_column": id_column,
                "fields": [], "issues": [], "field_edits": {}, "events": []}
    seen_ids = set()
    for row_index, row in sorted(latest.items()):
        report = row.get("qa_report", {})
        if row.get("qa_status") == "SKIPPED_NOT_HUMAN_CAPTION" or report.get("qa_scope") == "input_error":
            continue
        caption = row.get(caption_column)
        try:
            sections = qa.parse_caption_sections(caption)
        except ValueError:
            continue
        from caption_qa_width import normalize, valid_operation, unsafe_regions
        width_enabled=report.get('workflow_version')==AUTO_WORKFLOW
        normalized={name:normalize(name,value,exempt=name in qa.PUNCTUATION_EXEMPT_SECTIONS)
                    for name,value in sections.items()} if width_enabled else sections
        cleaned = qa.apply_static_fixes(normalized)
        proposal = report.get("proposed_caption", report.get("revised_caption", caption))
        try:
            proposed = qa.parse_caption_sections(proposal)
        except ValueError:
            proposed = sections
        identifier = row.get(id_column)
        identifier = str(identifier) if type(identifier) in (str, int) and str(identifier).strip() else ""
        if identifier and identifier in seen_ids:
            raise ValueError("题目 ID 重复，无法建立唯一审核记录；请核对原表。")
        if identifier:
            seen_ids.add(identifier)
        field_map = {}
        for name, value in sections.items():
            field_id = digest(f"{document['source_sha256']}:{row_index}:{name}")[:24]
            field_map[name] = {"field_id": field_id, "row_index": row_index, "sample_id": identifier,
                               "name": name, "original": value, "issue_ids": [],
                               "source_sheet": row.get("source_sheet"), "source_row": row.get("source_row")}
        items = [item for item in report.get("issues", [])
                 if item.get("source") in {"static", "llm"} and item.get("issue_type") not in DISABLED]
        if not items and report.get("semantic_qa_performed") and report.get("need_human_review"):
            reason = report.get("human_review_reason", "")
            if reason and not qa.DISABLED_FINDING_RE.search(reason):
                name = "自然指令" if "自然指令" in sections else next(iter(sections))
                items = [{"section": name, "issue_type": "其他疑似问题", "original_snippet": "",
                          "suggestion": reason, "can_auto_fix": False, "source": "llm"}]
        occurrence = Counter()
        row_issues = []
        for number, item in enumerate(items):
            name = item.get("section", "")
            if name not in sections:
                names = [n for n in name.split(" / ") if n in sections]
                name = next((n for n in names if item.get("original_snippet", "") in sections[n]), names[0] if names else "")
            if not name:
                continue
            field = field_map[name]
            snippet, kind = item.get("original_snippet", ""), item.get("issue_type", "")
            locations = (_locations(sections[name], normalized.get(name,""), snippet)
                         or _locations(sections[name], cleaned.get(name, ""), snippet))
            key = (name, kind, snippet)
            if item.get("source") == "static" and len(locations) > 1:
                locations = [locations[min(occurrence[key], len(locations) - 1)]]
                occurrence[key] += 1
            excerpt = list(dict.fromkeys(s for a, b in locations for s in sentence_context(sections[name], a, b)))
            related = (name == "主体" or kind in qa.HARD_TYPES or kind == "隐式漏挂主体标签"
                       or (kind in {"bare_tags", "裸标签漏写 <>"} and "主体" in snippet))
            if kind == "missing_subjects":
                excerpt = [line for line in sections.get("主体", "").splitlines() if snippet in line]
            if not excerpt:
                excerpt = [sections[name].strip()] if sections[name].strip() else []
            issue_id = digest(f"{field['field_id']}:{kind}:{snippet}:{number}")[:24]
            issue = {"issue_id": issue_id, "field_id": field["field_id"], "sample_id": identifier,
                     "row_index": row_index, "field": name, "kind": kind, "label": LABELS.get(kind, kind),
                     "snippet": snippet, "excerpts": excerpt, "subject": sections.get("主体", "").strip() if related else "",
                     "reason": item.get("suggestion", ""), "decision": "pending", "operations": [],
                     "locations": locations, "can_auto_fix": bool(item.get("can_auto_fix")), "origin": item.get("source"),
                     "source_issue_index": next((n for n, original in enumerate(report.get('issues', [])) if original is item), None)}
            if isinstance(item.get('replacement'),str):
                issue['replacement']=item['replacement']
            if width_enabled and kind=='symbol_width' and item.get('source')=='static' and item.get('width_operations'):
                issue['operations']=[dict(op) for op in item.get('width_operations',[])
                                     if valid_operation(name,sections[name],op,exempt=name in qa.PUNCTUATION_EXEMPT_SECTIONS)]
                issue['locations']=[(op['start'],op['end']) for op in issue['operations']]
                issue['excerpts']=list(dict.fromkeys(s for a,b in issue['locations'] for s in sentence_context(sections[name],a,b)))
                issue['excerpt_highlights']=[]
                for excerpt in issue['excerpts']:
                    spans=[]
                    for match in re.finditer(re.escape(excerpt),sections[name]):
                        spans.extend([a-match.start(),b-match.start()] for a,b in issue['locations']
                                     if match.start()<=a and b<=match.end())
                    issue['excerpt_highlights'].append(spans)
                if issue['subject']:
                    leading=len(sections[name])-len(sections[name].lstrip())
                    issue['subject_highlights']=[[a-leading,b-leading] for a,b in issue['locations']]
            if item.get('source') == 'static' and kind in {'bare_tags','missing_punctuation'}:
                for a,b in locations:
                    if kind == 'bare_tags':
                        raw=sections[name][a:b]
                        normalized_piece=normalize(name,raw) if width_enabled else raw
                        replacement = qa._normalize_tags(normalized_piece)
                        if replacement != normalized_piece and not touches_audio(normalized[name],a,b):
                            for code,x,y,u,v in difflib.SequenceMatcher(None,normalized_piece,replacement,autojunk=False).get_opcodes():
                                if code!='equal':
                                    issue['operations'].append({'start':a+x,'end':a+y,'old':raw[x:y],'new':replacement[u:v]})
                    elif not touches_audio(normalized[name],b,b):
                        issue['operations'].append({'start':b,'end':b,'old':'','new':'。'})
            if width_enabled:
                blocked=unsafe_regions(name,sections[name])
                issue['operations']=[op for op in issue['operations'] if not any(
                    (a<=op['start']<=b if op['start']==op['end'] else op['start']<b and op['end']>a) for a,b in blocked)]
            field["issue_ids"].append(issue_id)
            row_issues.append(issue)
        # Assign each validated proposed edit to exactly one eligible issue.
        # Approval of one issue must never authorize all other edits in a field.
        for name, field in field_map.items():
            old, new = sections[name], proposed.get(name, sections[name])
            static_operations = {(o['start'],o['end'],o['old'],o['new']) for i in row_issues
                                 if i['field_id'] == field['field_id'] and i['origin'] == 'static' for o in i['operations']}
            for op, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
                blocked=unsafe_regions(name,old) if width_enabled else []
                if (op == "equal" or touches_audio(normalized[name], a, b) or (a,b,old[a:b],new[c:d]) in static_operations
                        or any((x<=a<=y if a==b else a<y and b>x) for x,y in blocked)):
                    continue
                candidates = [i for i in row_issues if i["field_id"] == field["field_id"] and i["can_auto_fix"] and i['origin'] != 'static'
                              and any(x <= a and b <= y for x, y in i["locations"])]
                for issue in sorted(candidates, key=lambda i: (i["origin"] != "static", min(y - x for x, y in i["locations"]))):
                    kind = issue["kind"]
                    if not _eligible(old[a:b], new[c:d], kind, issue):
                        continue
                    if kind != "bare_tags":
                        check = dict(section=name, original_snippet=old, issue_type=kind, can_auto_fix=True)
                        if not qa._whitelisted_changes(old, old[:a] + new[c:d] + old[b:], [check]):
                            continue
                    issue["operations"].append({"start": a, "end": b, "old": old[a:b], "new": new[c:d]})
                    break
            # A model issue with only a partially executable proposal stays
            # wholly pending; never label an audio/protected remainder fixed.
            for issue in row_issues:
                if issue['field_id'] != field['field_id'] or issue['origin'] == 'static' or not issue['operations']:
                    continue
                expected = {(a,b,old[a:b],new[c:d]) for op,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes()
                            if op != 'equal' and (a,b,old[a:b],new[c:d]) not in static_operations
                            and any(x <= a and b <= y for x,y in issue['locations'])}
                actual = {(o['start'],o['end'],o['old'],o['new']) for o in issue['operations']}
                if not expected.issubset(actual):
                    issue['operations'] = []
        document["issues"].extend(row_issues)
        document["fields"].extend(field for field in field_map.values() if field["issue_ids"])
    from caption_qa_candidates import enrich_candidates
    enrich_candidates(document)
    if automatic:
        _approve_automatic(document)
    return document


def _approve_automatic(document):
    """Authorize only complete, non-conflicting validated operations."""
    for issue in document['issues']:
        if issue['sample_id'] and issue['operations'] and not issue.get('requires_confirmation'):
            issue['decision'] = 'auto'
    for field in document['fields']:
        selected = [i for i in document['issues'] if i['field_id'] == field['field_id'] and i['decision'] == 'auto']
        operations = sorted({(o['start'], o['end'], o['old'], o['new']) for i in selected for o in i['operations']})
        if any(right[0] < left[1] or right[0] == left[0] for left, right in zip(operations, operations[1:])):
            for issue in selected:
                issue['decision'] = 'pending'
    for issue in document['issues']:
        if issue['decision'] == 'auto':
            document['events'].append({'revision': 0, 'issue_id': issue['issue_id'], 'decision': 'auto',
                                       'policy': document['workflow_version']})
