"""Add current metadata/reference checks without replacing human decisions."""
import hashlib

POLICY = 'caption-metadata-ref-v2'


def enrich(document, records):
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_metadata import operations, confirmation_reason
    from captionqa.caption_qa_references import detect_columns, findings
    from captionqa.caption_qa_review_model import AUTO_WORKFLOWS
    columns = detect_columns(records.values())
    document['reference_columns'] = columns
    document['format_policy'] = POLICY
    # Keep only ref cells, never unrelated source content.
    keys={key for values in columns['media'].values() for key in values}
    if columns.get('counts'):keys.add(columns['counts'])
    document['reference_inventory']={str(index):{'record':{k:r.get(k) for k in keys}}
                                     for index,r in records.items() if qa.is_human_caption(r.get(document['caption_column']))}
    added = []
    existing = {(i['row_index'],i['field'],i['kind'],i['snippet']) for i in document['issues']}
    for index, record in records.items():
        caption = record.get(document['caption_column'])
        if not qa.is_human_caption(caption): continue
        try: sections = qa.parse_caption_sections(caption)
        except ValueError: continue
        proposals, _, notes = findings(sections, record, columns)
        # A saved human draft may already have removed an old bad reference.
        final_sections=dict(sections)
        for field in document['fields']:
            if field['row_index']==index and field['field_id'] in document['field_edits']:
                final_sections[field['name']]=document['field_edits'][field['field_id']]
        final_refs,_,_=findings(final_sections,record,columns)
        unresolved={(i['section'],i['original_snippet']) for i in final_refs}
        proposals=[i for i in proposals if (i['section'],i['original_snippet']) in unresolved]
        context = '输入描述' if '输入描述' in sections else next(iter(sections))
        proposals += [{'section':context,'issue_type':'reference_inventory','original_snippet':'',
                       'suggestion':'；'.join(notes)}] if notes else []
        for name, value in sections.items():
            ops = operations(name, value)
            if ops:
                proposals.append({'section':name,'issue_type':'tag_language','original_snippet':value.strip(),
                    'suggestion':'按明确词表将英文标签翻译为中文，保留原维度、相似关系与顺序。', 'operations':ops})
            reason=confirmation_reason(name,value)
            if reason:
                proposals.append({'section':name,'issue_type':'unknown_tag_language','original_snippet':value.strip(),
                    'suggestion':reason})
        for item in proposals:
            name, kind, snippet = item['section'], item['issue_type'], item['original_snippet']
            signature = (index,name,kind,snippet)
            if signature in existing:
                if kind=='unknown_tag_language':
                    previous=next(i for i in document['issues'] if (i['row_index'],i['field'],i['kind'],i['snippet'])==signature)
                    if previous.get('contract_policy') and previous['decision'] in {'pending','defer'} and previous['reason']!=item['suggestion']:
                        previous.update(reason=item['suggestion'],label='标签值待确认',contract_policy=POLICY)
                        added.append(previous['issue_id'])
                continue
            field = next((f for f in document['fields'] if f['row_index']==index and f['name']==name), None)
            if field and field['original'] != sections[name]: continue
            if field is None:
                identifier = record.get(document['id_column'])
                field = {'field_id':hashlib.sha256(f"{document['source_sha256']}:{index}:{name}".encode()).hexdigest()[:24],
                    'row_index':index,'sample_id':str(identifier) if type(identifier) in (str,int) else '',
                    'name':name,'original':sections[name],'issue_ids':[],
                    'source_sheet':record.get('source_sheet'),'source_row':record.get('source_row')}
                document['fields'].append(field)
            edits = item.get('operations', [])
            # No new rule may overlap already-authorized source edits.
            selected = [op for other in document['issues'] if other['field_id']==field['field_id']
                        and other['decision'] in {'auto','approve'} for op in other['operations']]
            if edits and any(a['start']<b['end'] and b['start']<a['end'] for a in edits for b in selected): continue
            automatic = bool(edits and field['sample_id'] and document['workflow_version'] in AUTO_WORKFLOWS
                             and field['field_id'] not in document['field_edits'])
            locations = [[op['start'],op['end']] for op in edits]
            if not locations and snippet:
                import re
                locations = [[m.start(),m.end()] for m in re.finditer(re.escape(snippet),sections[name])]
            key = hashlib.sha256(f"{field['field_id']}:{POLICY}:{kind}:{snippet}".encode()).hexdigest()[:24]
            issue = {'issue_id':key,'field_id':field['field_id'],'sample_id':field['sample_id'],
                'row_index':index,'field':name,'kind':kind,'label':{
                    'tag_language':'英文标签转中文','unknown_tag_language':'标签值待确认',
                    'reference_out_of_range':'参考素材引用越界','reference_inventory':'参考素材数量待确认'}[kind],
                'snippet':snippet,'excerpts':[snippet] if snippet else [],'subject':'',
                'reason':item['suggestion'],'decision':'auto' if automatic else 'pending',
                'operations':edits,'locations':locations,'can_auto_fix':bool(edits),
                'origin':'static','source_issue_index':None,'contract_policy':POLICY}
            document['issues'].append(issue); field['issue_ids'].append(key)
            existing.add(signature); added.append(key)
    return added
