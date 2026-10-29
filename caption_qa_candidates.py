"""Explicit local candidates are executable only after a human approval."""
import difflib
import re

POLICY = 'confirmed-typo-v1'
PUNCTUATION_POLICY = 'confirmed-width-v1'
EXPLICIT_PUNCTUATION_POLICY = 'confirmed-punctuation-v1'
REWRITE_POLICY = 'confirmed-rewrite-v1'
REWRITE_KINDS = {'语法病句与语句不通顺','隐式漏挂主体标签','内容符号不规范','裸标签漏写 <>',
                 '无歧义错别字','错别字与衍字'}
LOCAL_PUNCTUATION = frozenset(',，.。!！?？;；:：、…')
KINDS = {'无歧义错别字', '错别字与衍字'}
QUOTED = r'[“"「]([^”"」\n]{1,64})[”"」]'
# Do not disguise changed visual attributes, numbers or references as spelling.
FACT_CHARS = frozenset('红橙黄绿青蓝紫黑白灰棕银金左右前后上下内外东西南北长短宽窄厚薄大小高低男女一二三四五六七八九十百千万亿')


def _pair(issue):
    reason=issue.get('reason',issue.get('suggestion',''))
    snippet=issue.get('snippet',issue.get('original_snippet',''))
    if re.search(r'或者|或为|二选|两种|无法确定|不能确定|不确定|也可能|[”"」]\s*(?:或|/)',reason):
        return None
    if re.search(r'(?:不建议|不应|不宜|不要)\s*(?:将|改|替换)|无需修改|无须修改|不是错别字',reason):
        return None
    explicit=issue.get('replacement')
    if isinstance(explicit,str) and explicit.strip():
        return snippet,explicit.strip()
    pairs=[]
    # Older saved reports only have a natural-language suggestion.
    for match in re.finditer(QUOTED+r'\s*(?:疑似为|疑为|应为|应改为)\s*'+QUOTED,reason):
        old,new=match.group(1),match.group(2)
        if old in snippet:
            pairs.append((old,new))
    for match in re.finditer(r'(?:疑似应为|疑似为|疑为|应改为|应为|改成|改为)\s*'+QUOTED,reason):
        pairs.append((snippet,match.group(1)))
    valid=[]
    for old,new in dict.fromkeys(pairs):
        if _word_pair(old,new):valid.append((old,new))
    return valid[0] if len(valid)==1 else None


def _word_pair(old,new):
    if not (isinstance(old,str) and isinstance(new,str) and 1<=len(old)==len(new)<=32):
        return False
    if not re.fullmatch(r'[\u3400-\u9fff]+',old+new):
        return False
    differences=[(a,b) for a,b in zip(old,new) if a!=b]
    return len(differences)==1 and not any(c in FACT_CHARS for pair in differences for c in pair)


def candidate_operation(issue,original):
    if issue.get('kind',issue.get('issue_type')) not in KINDS:
        return None
    pair=_pair(issue)
    if not pair or not _word_pair(*pair):
        return None
    old,new=pair
    snippet=issue.get('snippet',issue.get('original_snippet',''))
    matches=list(re.finditer(re.escape(snippet),original)) if snippet else []
    # Ambiguous repeated locations require a human draft or better localization.
    if len(matches)!=1 or snippet.count(old)!=1:
        return None
    start=matches[0].start()+snippet.index(old);end=start+len(old)
    from caption_qa_review import touches_audio
    from caption_qa_symbols import LITERAL_RE
    from caption_qa_width import unsafe_regions
    field=issue.get('field',issue.get('section',''))
    if touches_audio(original,start,end) or any(m.start()<end and m.end()>start for m in LITERAL_RE.finditer(original)):
        return None
    if any(a<end and b>start for a,b in unsafe_regions(field,original)):
        return None
    # Reject an apparent spelling fix that is actually a duplicate deletion.
    if any(code in {'insert','delete'} for code,*_ in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes()):
        return None
    offset=next(index for index,(a,b) in enumerate(zip(old,new)) if a!=b)
    return {'start':start+offset,'end':start+offset+1,'old':old[offset],'new':new[offset],
            'candidate_old':old,'candidate_new':new,'review_only':True,'candidate_policy':POLICY}


def _explicit_punctuation_operations(issue,original):
    """A precise punctuation proposal may cross audio protection after approval.

    Parse a source/target pair, never infer the target from a nearby example.
    Content wrappers, words, whitespace and technical tokens remain opaque.
    """
    snippet=issue.get('snippet',issue.get('original_snippet',''))
    reason=issue.get('reason',issue.get('suggestion',''))
    if re.search(r'或者|或为|二选|两种|无法确定|不能确定|不确定|也可能|[”"」]\s*(?:或|/)',reason):
        return []
    if re.search(r'(?:不建议|不应|不宜|不要)\s*(?:将|把|改|替换)|无需修改|无须修改',reason):
        return []
    explicit=issue.get('replacement')
    if isinstance(explicit,str) and explicit:
        pairs=[(snippet,explicit)]
    else:
        # Supports 将“old”改为“new” and “old” → “new”, including braces
        # that give context but must themselves stay unchanged.
        pairs=[(m[1],m[2]) for m in re.finditer(
            QUOTED+r'\s*(?:改为|改成|替换为|替换成|应为|应改为|→|->)\s*'+QUOTED,reason)]
    pairs=list(dict.fromkeys(pairs))
    if len(pairs)!=1:
        return []
    old,new=pairs[0]
    if not old or old==new or snippet.count(old)!=1 or original.count(snippet)!=1:
        return []
    start=original.index(snippet)+snippet.index(old)
    from caption_qa_width import SCREEN, TECH, TAG, unsafe_regions
    from caption_qa_agent import PUNCTUATION_EXEMPT_SECTIONS
    field=issue.get('field',issue.get('section',''))
    if field in PUNCTUATION_EXEMPT_SECTIONS:
        return []
    protected=[m.span() for pattern in (SCREEN,TECH,TAG) for m in pattern.finditer(original)]
    # Numeric punctuation can encode values/timecodes even in non-ASCII form.
    protected.extend(m.span() for m in re.finditer(r'\d+(?:[.,，。:：;/／%％-]\d+)+',original))
    protected.extend(unsafe_regions(field,original))
    result=[]
    for code,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if code=='equal':
            continue
        before,after=old[a:b],new[c:d]
        if any(char not in LOCAL_PUNCTUATION for char in before+after):
            return []
        x,y=start+a,start+b
        if any((x<q and y>p) if x!=y else p<x<q for p,q in protected):
            return []
        # Do not create/remove a numeric separator where no token existed yet.
        if x>0 and y<len(original) and original[x-1].isdigit() and original[y].isdigit():
            return []
        result.append({'start':x,'end':y,'old':before,'new':after,
                       'candidate_old':old,'candidate_new':new,'review_only':True,
                       'candidate_policy':EXPLICIT_PUNCTUATION_POLICY})
    return result


def _restricted_operations(issue,original):
    if issue.get('kind',issue.get('issue_type')) in KINDS:
        operation=candidate_operation(issue,original)
        return [operation] if operation else []
    if issue.get('kind',issue.get('issue_type')) not in {'标点符号硬伤','标点与标签格式'}:
        return []
    snippet=issue.get('snippet',issue.get('original_snippet',''))
    matches=list(re.finditer(re.escape(snippet),original)) if snippet else []
    if len(matches)!=1:
        return []
    reason=issue.get('reason',issue.get('suggestion',''))
    explicit=issue.get('replacement')
    fallback=lambda:_explicit_punctuation_operations(issue,original)
    if not explicit and not ('半角' in reason and '全角' in reason):
        return fallback()
    if re.search(r'或者|二选|无法确定|不建议|不应|不要',reason):
        return []
    from caption_qa_width import operations
    from caption_qa_review import touches_audio
    from caption_qa_symbols import LITERAL_RE
    from caption_qa_agent import PUNCTUATION_EXEMPT_SECTIONS
    field=issue.get('field',issue.get('section',''))
    a,b=matches[0].span()
    markers=[('冒号',':'),('逗号',','),('分号',';'),('问号','?'),('叹号','!'),('句号','.'),('括号','()')]
    allowed=''.join(chars for word,chars in markers if word in reason)
    result=[]
    for operation in operations(field,original,exempt=field in PUNCTUATION_EXEMPT_SECTIONS):
        x,y=operation['start'],operation['end']
        if not (a<=x and y<=b) or (not explicit and operation['old'] not in allowed):
            continue
        outer=operation['width_rule'] in {'背景音乐外层括号','人声包裹符','音效包裹符'}
        prefix=any(m.start('colon')==x for m in re.finditer(r'(?m)^[ \t]*[<＜]主体\d+[>＞](?P<colon>:)',original))
        if touches_audio(original,x,y) and not (outer or prefix):
            continue
        if any(m.start()<y and m.end()>x for m in LITERAL_RE.finditer(original)) and not outer:
            continue
        result.append(dict(operation,review_only=True,candidate_policy=PUNCTUATION_POLICY))
    if explicit and result:
        proposed=snippet
        for operation in reversed(result):
            x,y=operation['start']-a,operation['end']-a
            proposed=proposed[:x]+operation['new']+proposed[y:]
        if proposed!=explicit:
            return fallback()
    if not result:
        # A human can approve a quoted width-only change inside a Chinese
        # sound-effect description. Default audio protection remains intact.
        from caption_qa_width import ANGLE, SOUND, TAG
        sound=any(m.start()<a and b<m.end() and SOUND.search(m[0]) and not TAG.fullmatch(m[0])
                  for m in ANGLE.finditer(original))
        if sound:
            quoted=re.findall(QUOTED,reason)
            inner=[o for o in operations('确认标点',snippet) if o['width_rule']=='中文叙述标点'
                   and (explicit or o['old'] in allowed)]
            proposed=snippet
            for operation in reversed(inner):
                proposed=proposed[:operation['start']]+operation['new']+proposed[operation['end']:]
            if inner and proposed in ([explicit] if explicit else quoted):
                result=[dict(operation,start=operation['start']+a,end=operation['end']+a,
                             review_only=True,confirmed_audio_width=True,candidate_policy=PUNCTUATION_POLICY)
                        for operation in inner]
    return result or fallback()


def _rewrite_operations(issue,original):
    """Turn an exact suggested rewrite into a proposal, never an auto fix."""
    kind=issue.get('kind',issue.get('issue_type'))
    if kind not in REWRITE_KINDS:
        return []
    snippet=issue.get('snippet',issue.get('original_snippet',''))
    reason=issue.get('reason',issue.get('suggestion',''))
    from caption_qa_suggestions import _ambiguous
    if _ambiguous(reason):
        return []
    from caption_qa_agent import DISABLED_FINDING_RE, PUNCTUATION_EXEMPT_SECTIONS, _disabled_finding, _normalize_tags
    field=issue.get('field',issue.get('section',''))
    if (field in PUNCTUATION_EXEMPT_SECTIONS or
        DISABLED_FINDING_RE.search(reason) or
        re.search(r'或者|或为|二选|两种|无法确定|不能确定|不确定|也可能|[”"」]\s*(?:或|/)',reason) or
        re.search(r'(?:不建议|不应|不宜|不要)\s*(?:将|把|改|替换)|无需修改|无须修改',reason)):
        return []
    quoted=r'[“"「]([^”"」\n]{1,4000})[”"」]'
    explicit=issue.get('replacement')
    if isinstance(explicit,str) and explicit:
        pairs=[(snippet,explicit)]
    else:
        pairs=[(m[1],m[2]) for m in re.finditer(
            quoted+r'\s*(?:改为|改成|替换为|替换成|应改为|→|->)\s*'+quoted,reason)]
        if not pairs:
            pairs=[(snippet,m[1]) for m in re.finditer(
                r'(?:例如|比如|如|建议(?:修改|改写|重写)?为|(?:修改|改写|重写|改成|改)为(?:通顺表述)?)[：:\s]*'+quoted,reason)]
    pairs=list(dict.fromkeys(pairs))
    if len(pairs)!=1:
        return []
    old,new=pairs[0]
    if (not old or old==new or len(new)>4000 or len(new)==1 and len(old)>3 or '/' in new and '/' not in old or
        snippet.count(old)!=1 or original.count(snippet)!=1 or
        re.search(r'\r|\n|##|……|\.\.\.',new)):
        return []
    # A sentence rewrite may reorder existing labels, but must not renumber
    # references, change times/numbers or invent a new reference. Subject-link
    # proposals can explicitly add a label for a human to verify.
    from collections import Counter
    from caption_qa_width import TAG, TECH, unsafe_regions
    if _disabled_finding({'issue_type':kind,'suggestion':reason,'section':field,'original_snippet':old},
                         {field:old},{field:new}):
        return []
    changes=[(code,a,b,c,d) for code,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes() if code!='equal']
    if changes and all(code=='delete' and (old[max(0,a-(b-a)):a]==old[a:b] or old[b:b+(b-a)]==old[a:b])
                       for code,a,b,c,d in changes):
        return []
    if kind in KINDS and Counter(c for c in old if c in FACT_CHARS)!=Counter(c for c in new if c in FACT_CHARS):
        return []
    tags=lambda value:Counter(m[0] for m in TAG.finditer(value))
    if kind not in {'隐式漏挂主体标签','裸标签漏写 <>'} and tags(old)!=tags(new):
        return []
    if kind=='裸标签漏写 <>' and _normalize_tags(old)!=new:
        return []
    numeric=lambda value:re.findall(r'\d+(?:\.\d+)?',TAG.sub('',_normalize_tags(value)))
    if numeric(old)!=numeric(new):
        return []
    from caption_qa_review import touches_audio
    from caption_qa_symbols import LITERAL_RE, SYMBOL_DELIMITERS
    technical=[m.span() for m in TECH.finditer(original)]
    protected=[m.span() for m in LITERAL_RE.finditer(original)]
    blocked=unsafe_regions(field,original)
    start=original.index(snippet)+snippet.index(old)
    result=[]
    for code,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if code=='equal':
            continue
        x,y=start+a,start+b
        overlaps=lambda spans:any((x<q and y>p) if x!=y else p<x<q for p,q in spans)
        wrapper_change=(kind=='内容符号不规范' and all(char in SYMBOL_DELIMITERS|set('()＜＞｛｝') for char in old[a:b]+new[c:d]))
        touches_technical=any((x<q and y>p) if x!=y else p<=x<=q for p,q in technical)
        if (touches_technical or (overlaps(blocked+protected) or touches_audio(original,x,y)) and not wrapper_change):
            return []
        result.append({'start':x,'end':y,'old':old[a:b],'new':new[c:d],
                       'candidate_old':old,'candidate_new':new,'review_only':True,'candidate_policy':REWRITE_POLICY})
    return result


def candidate_operations(issue,original):
    # Preserve saved narrow candidates exactly, so existing approvals remain
    # verifiable when the human-confirmation scope expands.
    from caption_qa_suggestions import operations
    if issue.get('suggestion_candidate'):
        return operations(issue,original)
    return _restricted_operations(issue,original) or _rewrite_operations(issue,original) or operations(issue,original)


def enrich_candidates(document):
    """Fill empty proposals without changing any existing human decision."""
    fields={field['field_id']:field for field in document['fields']}
    count=0
    for issue in document['issues']:
        if issue.get('operations') or issue.get('origin')!='llm' or issue['decision'] not in {'pending','defer'}:
            continue
        if issue['field_id'] in document.get('field_edits',{}):
            continue
        field=fields[issue['field_id']]
        operations=candidate_operations(issue,field['original'])
        if operations:
            issue['operations']=operations
            issue['requires_confirmation']=True
            issue['candidate_policy']=operations[0]['candidate_policy']
            count+=1
    return count


def valid_candidate(issue,original,operation):
    return any(all(expected.get(key)==operation.get(key) for key in expected)
               for expected in candidate_operations(issue,original))
