"""Pure review decisions and provenance; no model, table or HTTP access."""
from collections import Counter
import copy
from caption_qa_review_model import DECISIONS


def automatic_issue_ids(document):
    """Automatic provenance stays hidden even after a whole-field human draft."""
    return ({i['issue_id'] for i in document['issues'] if i['decision']=='auto'} |
            {event['issue_id'] for event in document.get('events',[]) if event.get('decision')=='auto' and event.get('issue_id')})


def human_review_issues(document):
    automatic=automatic_issue_ids(document)
    return [issue for issue in document['issues'] if issue['issue_id'] not in automatic]


def highlighted_rows(document):
    """Keep just the chosen issue; mark its Caption, without canceling other edits."""
    rows={}
    for issue in human_review_issues(document):
        if issue['decision']!='keep_highlight':continue
        if not issue['sample_id']:raise ValueError('Caption标黄需要原表真实题目ID。')
        row=rows.setdefault(issue['row_index'],{'sample_id':issue['sample_id'],'issue_ids':[],'fields':[]})
        row['issue_ids'].append(issue['issue_id'])
        if issue['field'] not in row['fields']:row['fields'].append(issue['field'])
    return rows


def summary(document):
    counts = Counter(item["decision"] for item in document["issues"])
    visible=human_review_issues(document)
    pending=[i for i in visible if i['decision'] in {'pending','defer'}]
    return {"revision": document["revision"], "issues": len(document["issues"]),
            "pending": len(pending), "human_issues":len(visible),
            "needs_confirmation":sum(bool(i['operations']) for i in pending),
            "needs_judgment":sum(not i['operations'] for i in pending), "decisions": dict(counts),
            "manual_fields": len(document["field_edits"]), "automatic_issues": counts['auto'],
            "source_sha256": document["source_sha256"],"highlighted_captions":len(highlighted_rows(document))}


def decide(document, *, revision, issue_id=None, decision=None, field_id=None, final_text=None, reset=False):
    if type(revision) is not int or revision != document["revision"]:
        raise ValueError("审核版本已变化，请刷新后重试。")
    result = copy.deepcopy(document)
    issues = {i["issue_id"]: i for i in result["issues"]}
    fields = {f["field_id"]: f for f in result["fields"]}
    event = {"revision": revision + 1}
    if issue_id is not None:
        if issue_id not in issues or decision not in DECISIONS - {"manual", "auto"}:
            raise ValueError("错误项或审核结论无效。")
        issue = issues[issue_id]
        if issue["field_id"] in result["field_edits"]:
            raise ValueError("此字段已有人工定稿，请先撤销定稿再调整单项结论。")
        if decision == "approve" and (not issue["sample_id"] or not issue["operations"]):
            raise ValueError("此项没有可安全执行的指定改法，或缺少真实题目 ID；请人工填写完整字段。")
        if decision=='keep_highlight' and (not issue['sample_id'] or issue_id in automatic_issue_ids(result)):
            raise ValueError('请仅对具备真实题目ID的人工审核项选择保留原文并标黄。')
        issue["decision"] = decision
        event.update(issue_id=issue_id, decision=decision)
    elif field_id in fields:
        field = fields[field_id]
        if reset:
            result["field_edits"].pop(field_id, None)
            for key in field["issue_ids"]:
                if issues[key]["decision"] == "manual":
                    issues[key]["decision"] = "pending"
            event.update(field_id=field_id, decision="reset_manual")
        else:
            if not field["sample_id"] or not isinstance(final_text, str) or not final_text.strip() or len(final_text) > 100000:
                raise ValueError("人工稿须为非空的完整字段，并具备真实题目 ID。")
            result["field_edits"][field_id] = final_text
            for key in field["issue_ids"]:
                if issues[key]["decision"] not in {"reject", "keep", "keep_highlight"}:
                    issues[key]["decision"] = "manual"
            event.update(field_id=field_id, decision="manual", before=field["original"], after=final_text)
    else:
        raise ValueError("请选择有效的错误项或字段。")
    result["revision"] += 1
    result["events"].append(event)
    return result
