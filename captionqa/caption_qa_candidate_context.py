"""Keep an inserted punctuation mark from duplicating adjacent source text."""
import re
from copy import deepcopy

from captionqa.caption_qa_review_audio import touches_audio
from captionqa.caption_qa_width import SCREEN


def contextual_operations(original, operations):
    result = []
    removed = []
    for raw in operations:
        op = dict(raw)
        p = op['start']
        insert = op['new']
        protected = touches_audio(original, p, p) or any(m.start()<p<m.end() for m in SCREEN.finditer(original))
        # Only newly inserted, single Chinese punctuation marks are redundant.
        # Existing punctuation and expressive !! / ... are never collapsed.
        left,right=original[:p],original[p:]
        for other in sorted(operations,key=lambda o:(o['start'],o['end']),reverse=True):
            if other['start']==other['end']==p:
                continue
            if other['end']<=p:
                left=left[:other['start']]+other['new']+left[other['end']:]
            elif other['start']>=p:
                a,b=other['start']-p,other['end']-p
                right=right[:a]+other['new']+right[b:]
        if (op['start']==op['end'] and not op['old'] and re.fullmatch(r'[。，、；：]', insert)
                and not protected and (right[:1]==insert or left[-1:]==insert)):
            removed.append(op)
        else:
            result.append(op)
    if not removed:
        return result
    # Rebuild preview text from effective operations at the original anchor.
    # This makes the displayed candidate identical to what export executes.
    effective=deepcopy(result)
    for op in result:
        old = op.get('candidate_old')
        if not old:
            continue
        starts = [m.start() for m in re.finditer(re.escape(old), original)
                  if m.start()<=op['start'] and op['end']<=m.end()]
        if len(starts)!=1:
            continue
        start=starts[0];end=start+len(old);new=old
        related=[o for o in effective if start<=o['start']<=o['end']<=end
                 and o.get('candidate_old')==old and o.get('candidate_new')==op.get('candidate_new')]
        for change in sorted(related,key=lambda o:(o['start'],o['end']),reverse=True):
            a,b=change['start']-start,change['end']-start
            new=new[:a]+change['new']+new[b:]
        op['candidate_new']=new
    return result


def normalized_candidate(issue, original):
    """Upgrade only reproducible candidates; never authorize a stale proposal."""
    from captionqa.caption_qa_candidates import raw_candidate_operations
    result=deepcopy(issue)
    if not issue.get('requires_confirmation'):
        return result
    raw=raw_candidate_operations(issue,original)
    if raw != issue['operations']:
        return result
    ops=contextual_operations(original,raw)
    result['operations']=ops
    if raw and not ops:
        result['candidate_satisfied']=True
    return result


def normalize_saved_candidates(document):
    fields={f['field_id']:f for f in document['fields']}
    changed=[]
    for issue in document['issues']:
        updated=normalized_candidate(issue,fields[issue['field_id']]['original'])
        if updated != issue:
            issue.update(updated);changed.append(issue['issue_id'])
    return changed
