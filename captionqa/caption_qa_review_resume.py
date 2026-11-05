"""Carry source-bound human review work across a partial checkpoint retry."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import shutil
from uuid import uuid4


def prepare_retry(store):
    """Persist a recovery marker before a retry can append to its checkpoint."""
    from captionqa.caption_qa_review_store import atomic_json
    marker=store.cache.with_suffix('.review_resume.json')
    if marker.exists():return
    if not store.path.exists():return
    document=store.load()  # Validate all source/checkpoint bindings first.
    nonce=uuid4().hex
    archive=store.path.with_name(store.path.stem+'_previous_'+nonce[:8]+'.json')
    shutil.copyfile(store.path,archive)
    atomic_json(marker,{'version':1,'resume_id':nonce,'document':document,'archive':archive.name})


def resume_review(store):
    """Rebuild findings, then retain explicit decisions on unchanged fields.

    Full-field human drafts are authoritative even if the retried model now
    reports no finding for that field. Original source bodies and real IDs
    must still match. Nothing is inferred from a similar-looking caption.
    """
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_review import make_review
    from captionqa.caption_qa_review_store import source_records,validate_binding
    marker=store.cache.with_suffix('.review_resume.json')
    if not marker.exists():return
    saved=json.loads(marker.read_text())
    if saved.get('version')!=1:raise ValueError('未知的重试审核恢复版本，原审核备份保留。')
    old=saved['document']
    for key,path in [('source_sha256',store.source),('input_sha256',store.input_source)]:
        if old[key]!=qa._source_sha256(path):
            raise ValueError('重试时来源文件发生变化，未套用旧人工稿；原审核及备份保留。')
    if store.path.exists():
        current=json.loads(store.path.read_text())
        if current.get('retry_resume',{}).get('resume_id')==saved['resume_id']:
            if current['cache_sha256']==qa._source_sha256(store.cache):
                validate_binding(current,store.source,store.input_source,store.cache)
                marker.unlink(missing_ok=True)
                return
            # A prior save may have completed before marker cleanup. If a
            # subsequent retry appended more results, carry the newer review,
            # rather than restoring the older snapshot in the marker.
            if any(current.get(k)!=old.get(k) for k in
                   ('source_sha256','input_sha256','caption_column','id_column','sheet')):
                raise ValueError('恢复标记与当前审核来源不一致，原审核及备份保留。')
            old=current
    fresh=make_review(store.cache,store.source,store.input_source,
        caption_column=old['caption_column'],id_column=old['id_column'],sheet=old['sheet'])
    records=source_records(old,store.input_source)
    fields={f['field_id']:f for f in fresh['fields']}
    old_fields={f['field_id']:f for f in old['fields']}
    retained=[i for i in old['issues'] if i['decision'] in
              {'approve','reject','keep','keep_highlight','manual','defer'}]
    wanted=set(old.get('field_edits',{})) | {i['field_id'] for i in retained}
    for key in old_fields:
        if key not in wanted:continue
        field=old_fields[key];record=records.get(field['row_index'],{})
        identifier=record.get(old['id_column'])
        text=record.get(old['caption_column'])
        try:body=qa.parse_caption_sections(text).get(field['name'])
        except ValueError:body=None
        if body!=field['original'] or str(identifier)!=field['sample_id']:
            raise ValueError('重试后的字段原文或题目ID与人工稿不一致，未覆盖；原审核备份保留。')
        if key in fields and (fields[key]['original']!=body or fields[key]['sample_id']!=field['sample_id']):
            raise ValueError('重试检测字段与原表不一致，未覆盖人工稿。')
        if key not in fields:
            field=deepcopy(field);field['issue_ids']=[]
            fresh['fields'].append(field);fields[key]=field

    def signature(issue):
        return (issue['field_id'],issue['kind'],issue['snippet'],issue.get('origin'),
                tuple(tuple(span) for span in issue.get('locations',[])))
    slots=defaultdict(list)
    for index,issue in enumerate(fresh['issues']):slots[signature(issue)].append(index)
    for issue in retained:
        matches=slots[signature(issue)]
        copied=deepcopy(issue);copied['retained_after_retry']=True
        if matches:fresh['issues'][matches.pop(0)]=copied
        else:fresh['issues'].append(copied)
    fresh['field_edits']=deepcopy(old.get('field_edits',{}))
    # A full-field draft supersedes even newly discovered findings in that
    # field. Preserve explicit keeps/rejects, and let final checks warn.
    for issue in fresh['issues']:
        if issue['field_id'] in fresh['field_edits'] and issue['decision'] not in {'keep','keep_highlight','reject'}:
            issue['decision']='manual'
    for field in fresh['fields']:field['issue_ids']=[]
    ids={i['issue_id'] for i in fresh['issues'] if i.get('retained_after_retry')}
    reserved={i['issue_id'] for i in fresh['issues']}
    renamed={}
    # Report order can change on retry. Keep confirmed IDs stable; re-key a
    # fresh finding if its ordinal-based ID now collides with a retained one.
    for index,issue in enumerate(fresh['issues']):
        if issue.get('retained_after_retry'):continue
        previous=issue['issue_id']
        if previous in ids:
            candidate=hashlib.sha256(f"{saved['resume_id']}:{previous}:{index}".encode()).hexdigest()[:24]
            while candidate in reserved:candidate=hashlib.sha256(candidate.encode()).hexdigest()[:24]
            issue['issue_id']=candidate;renamed[previous]=candidate;reserved.add(candidate)
        ids.add(issue['issue_id'])
    ids=set()
    for issue in fresh['issues']:
        if issue['issue_id'] in ids:
            raise ValueError('重试审核项标识冲突，原审核备份保留，未静默丢弃修改。')
        ids.add(issue['issue_id']);fields[issue['field_id']]['issue_ids'].append(issue['issue_id'])
    fresh['revision']=old['revision']+1
    existing_auto={e.get('issue_id') for e in old.get('events',[]) if e.get('decision')=='auto'}
    final_automatic={i['issue_id'] for i in fresh['issues'] if i['decision']=='auto'}
    for event in fresh.get('events',[]):
        if event.get('issue_id') in renamed:event['issue_id']=renamed[event['issue_id']]
    fresh['events']=deepcopy(old.get('events',[]))+[
        e for e in fresh.get('events',[]) if e.get('decision')!='auto' or
        (e.get('issue_id') in final_automatic and e.get('issue_id') not in existing_auto)]
    fresh['retry_resume']={'resume_id':saved['resume_id'],
        'manual_fields':len(fresh['field_edits']),'retained_decisions':len(retained),'archive':saved['archive']}
    fresh['events'].append({'revision':fresh['revision'],'decision':'retry_review_resumed',
        **fresh['retry_resume'],'reason':'仅承接原文及题目ID未变化的人工稿和审核决定。'})
    store.save(fresh)
    marker.unlink(missing_ok=True)
