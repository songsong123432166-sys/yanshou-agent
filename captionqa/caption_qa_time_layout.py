"""Timeline layout from explicit camera cuts; audio and literal text are opaque."""
import hashlib
import re

POLICY='time-cut-boundary-v3'
PREVIOUS_POLICY='time-cut-layout-v2'
LEGACY_POLICY='time-range-paragraph-v1'
RANGE=re.compile(r'(?<![\w.<>])(?:第[ \t]*)?\d+(?:\.\d+)?[ \t]*(?:秒[ \t]*)?'
    r'[-~～—–至到][ \t]*\d+(?:\.\d+)?[ \t]*(?:秒|s\b)(?=[ \t]*(?:[，,:：]|$))',re.I)
POINT=re.compile(r'第[ \t]*\d+(?:\.\d+)?[ \t]*秒(?=[ \t]*(?:[，,:：]|$))')
CUT=re.compile(r'镜头(?:快速|迅速|直接|再次|突然|随即|立即)?切(?:换|至|到)|切镜')
MARKER='镜头多次切换'
REASON='时间段间空一行；段内切镜不足3次时仅在明确切镜处换行，其他“第X秒”不强制换行；至少3次时标注“镜头多次切换”。'


def _context(text):
    from captionqa.caption_qa_review_audio import audio_spans
    from captionqa.caption_qa_width import SCREEN, TECH
    matches=list(RANGE.finditer(text))
    technical=[m.span() for m in TECH.finditer(text)
               if not any(r.start()<=m.start() and m.end()<=r.end() for r in matches)]
    protected=audio_spans(text)+[m.span() for m in SCREEN.finditer(text)]+technical
    protected += [m.span() for m in re.finditer(r'“[^”]*”|「[^」]*」|"[^"\n]*"',text)]
    masked=list(text)
    for a,b in protected:
        masked[a:b]=' '*len(masked[a:b])
    return ''.join(masked),[m for m in matches if not any(a<=m.start()<b for a,b in protected)]


def legacy_operations(field,text):
    """Keep previously approved v1 whitespace operations verifiable."""
    if field!='自然指令':return []
    masked,matches=_context(text)
    newline='\r\n' if '\r\n' in text else '\n'
    result=[]
    for match in matches:
        gap=re.search(r'[ \t\r\n]*$',text[:match.start()])
        start=gap.start()
        if not text[:start].strip() or gap[0].count('\n')>=2:continue
        result.append({'start':start,'end':match.start(),'old':gap[0],'new':newline*2,
                       'time_range_rule':LEGACY_POLICY,'heading':match[0]})
    return result


def _negated(text,start):
    return bool(re.search(r'(?:不|未|无|没有|无需|并非|禁止|避免)(?:发生|进行|再|需|要)?[ \t]*$',text[max(0,start-10):start]))


def _operations(field,text,legacy_points=False):
    if field!='自然指令':return []
    masked,ranges=_context(text)
    newline='\r\n' if '\r\n' in text else '\n'
    result=[];policy=PREVIOUS_POLICY if legacy_points else POLICY
    for op in legacy_operations(field,text):
        result.append({**op,'time_range_rule':policy,'layout_kind':'range_break','anchor':op['end']})
    blocks=[(m.start(),m.end(),ranges[n+1].start() if n+1<len(ranges) else len(text),m)
            for n,m in enumerate(ranges)]
    if not ranges or masked[:ranges[0].start()].strip():
        blocks.insert(0,(0,0,ranges[0].start() if ranges else len(text),None))
    for start,body,end,heading in blocks:
        cuts=[m for m in CUT.finditer(masked,body,end) if not _negated(masked,m.start())
              and masked[body:m.start()].strip(' \t\r\n，,:：。；;')]
        declared=any(not _negated(masked,m.start()) for m in re.compile(MARKER).finditer(masked,body,end))
        if declared or len(cuts)>=3:
            if not declared and heading is not None:
                separator=re.match(r'[ \t]*[，,:：][ \t]*',text[body:end])
                insert=body+(separator.end() if separator else 0)
                result.append({'start':insert,'end':insert,'old':'','new':MARKER+'，',
                    'time_range_rule':policy,'layout_kind':'multi_cut_marker','heading':heading[0],'anchor':start})
            continue
        points=list(POINT.finditer(masked,body,end))
        boundaries=points if legacy_points else []
        if not legacy_points:
            for cut in cuts:
                prior=[p for p in points if p.end()<=cut.start() and not re.search(r'[。！？!?；;\r\n]',masked[p.end():cut.start()])]
                boundary=prior[-1] if prior else cut
                if not boundaries or boundaries[-1].start()!=boundary.start():boundaries.append(boundary)
        for point in boundaries:
            # An adjacent repeated start time is part of the opening timestamp,
            # e.g. "0-4秒，第0秒，...", rather than a later event boundary.
            if (heading is not None and not masked[body:point.start()].strip(' \t\r\n，,:：')
                    and float(re.search(r'\d+(?:\.\d+)?',heading[0])[0])==float(re.search(r'\d+(?:\.\d+)?',point[0])[0])):continue
            gap=re.search(r'[ \t\r\n]*$',text[:point.start()])
            if not text[:gap.start()].strip() or '\n' in gap[0]:continue
            result.append({'start':gap.start(),'end':point.start(),'old':gap[0],'new':newline,
                'time_range_rule':policy,'layout_kind':'point_break' if legacy_points else 'cut_break','heading':point[0],'anchor':point.start()})
    return sorted(result,key=lambda op:(op['start'],op['end']))


def operations(field,text):return _operations(field,text)


def previous_operations(field,text):return _operations(field,text,legacy_points=True)


def findings(field,text):
    grouped={}
    for op in operations(field,text):
        anchor=op['anchor'];grouped.setdefault(anchor,{'anchor':anchor,'heading':op['heading'],'operations':[]})['operations'].append(op)
    return list(grouped.values())


def normalize(field,text):
    for op in reversed(operations(field,text)):
        text=text[:op['start']]+op['new']+text[op['end']:]
    return text


def normalize_selected(field,original,text,selected):
    """Compose approved layout with semantic edits without applying kept findings."""
    def keys(source,ops):
        masked,ranges=_context(source)
        available={'range_break':ranges,'multi_cut_marker':ranges,'cut_break':list(POINT.finditer(masked))+list(CUT.finditer(masked))}
        result=set()
        for op in ops:
            kind=op.get('layout_kind','range_break');anchor=op.get('anchor',op['end'])
            if kind=='point_break':kind='cut_break'
            same=sorted((m for m in available[kind] if m[0]==op['heading']),key=lambda m:m.start())
            for number,match in enumerate(same):
                if match.start()==anchor:result.add((kind,match[0],number))
        return result
    wanted=keys(original,selected)
    for op in reversed(operations(field,text)):
        if keys(text,[op]) & wanted:text=text[:op['start']]+op['new']+text[op['end']:]
    return text


def enrich_time_layout(document,records):
    """Add the new local rule to old reviews; preserve explicit human choices."""
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_review_model import AUTO_WORKFLOWS
    automatic=document['workflow_version'] in AUTO_WORKFLOWS
    changed=[]
    for index,record in records.items():
        caption=record.get(document['caption_column'])
        if not isinstance(caption,str) or not qa.is_human_caption(caption):continue
        try:sections=qa.parse_caption_sections(caption)
        except ValueError:continue
        original=sections.get('自然指令','')
        ops=operations('自然指令',original)
        if not ops:continue
        field=next((f for f in document['fields'] if f['row_index']==index and f['name']=='自然指令'),None)
        if field and (field['field_id'] in document.get('field_edits',{}) or field['original']!=original):continue
        if field is None:
            raw_id=record.get(document['id_column'])
            sample_id=str(raw_id) if type(raw_id) in (str,int) else ''
            field={'field_id':hashlib.sha256(f"{document['source_sha256']}:{index}:自然指令".encode()).hexdigest()[:24],
                'row_index':index,'sample_id':sample_id,'name':'自然指令','original':original,'issue_ids':[],
                'source_sheet':record.get('source_sheet'),'source_row':record.get('source_row')}
            document['fields'].append(field)
        known={('cut_break' if o.get('layout_kind')=='point_break' else o.get('layout_kind','range_break'),o.get('anchor',o['end']))
            for i in document['issues'] if i['field_id']==field['field_id'] and i['kind']=='time_range_break'
            for o in i['operations']}
        for group in findings('自然指令',original):
            pending=[o for o in group['operations'] if (o['layout_kind'],o['anchor']) not in known]
            if not pending:continue
            anchor=group['anchor'];heading=group['heading']
            key=hashlib.sha256(f"{field['field_id']}:{POLICY}:{anchor}".encode()).hexdigest()[:24]
            decision='auto' if automatic and field['sample_id'] else 'pending'
            issue={'issue_id':key,'field_id':field['field_id'],'sample_id':field['sample_id'],
                'row_index':index,'field':'自然指令','kind':'time_range_break','label':'时间段分段',
                'snippet':heading,'excerpts':[original[max(0,anchor-35):anchor+70]],'subject':'',
                'reason':REASON,
                'decision':decision,'operations':pending,'locations':[[anchor,anchor+len(heading)]],
                'can_auto_fix':True,'origin':'static','source_issue_index':None}
            document['issues'].append(issue);field['issue_ids'].append(key);changed.append(key)
    return changed
