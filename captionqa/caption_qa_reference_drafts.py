"""Broader, source-bound reference drafts, executable only after human approval.

This policy is used only by the interactive proposal button. It does not widen
automatic fixes or reinterpret previously saved candidates.
"""
from collections import Counter
import difflib
import re

POLICY = 'confirmed-reference-draft-v2'
TAG = re.compile(r'[<＜]\s*(主体|场景|图片|视频|音频|画面元素)\s*(\d+)\s*[>＞]')
MOTION = re.compile(r'向前|向后|向左|向右|左侧|右侧|左手|右手|左脚|右脚|左脸|右脸')


def reference_tags(sections):
    """Existing definitions/material references, not invented model IDs."""
    import captionqa.caption_qa_agent as qa
    tags=set(qa._subject_definitions(sections.get('主体','')))
    for name,prefix in [('场景','场景'),('画面元素','画面元素')]:
        tags.update(re.findall(r'(?m)^\s*(<'+prefix+r'\d+>)\s*[：:]',sections.get(name,'')))
    for text in sections.values():
        tags.update(m[0] for m in TAG.finditer(text) if m[1] in {'图片','视频','音频'})
    return sorted(tags)


def editable_ranges(issue,original):
    """Allow the current paragraph, instead of forcing edits inside red text.

    A short red snippet is a locator, not a boundary for grammatical repairs.
    Multi-paragraph source windows still cannot authorize edits on other lines.
    """
    snippet=issue.get('snippet','')
    places=[m.span() for m in re.finditer(re.escape(snippet),original)] if snippet else []
    if not places:
        places=[(a,b) for a,b in issue.get('locations',[]) if 0<=a<=b<=len(original)]
    if not places:
        return []
    return sorted(set((original.rfind('\n',0,a)+1,
        original.find('\n',b) if original.find('\n',b)>=0 else len(original)) for a,b in places))


def _lexical(text):
    return ''.join(c for c in text if c.isalnum())


def draft_operations(issue,original):
    """Validate a model's recommendation, with no automatic-fix whitelist.

    Grammar/style may be rewritten within the located paragraph. Source IDs,
    timing, numeric facts, technical literals and quoted screen/speech words
    remain bound to the source. Semantic correctness is still a human choice.
    """
    from captionqa.caption_qa_summary import DISABLED
    import captionqa.caption_qa_agent as qa
    from captionqa.caption_qa_width import TECH, SCREEN
    from captionqa.caption_qa_review_audio import audio_spans
    proposal=issue.get('suggestion_candidate') or {}
    if proposal.get('policy')!=POLICY or issue.get('kind') in DISABLED:
        return []
    old,new,start=proposal.get('source'),proposal.get('replacement'),proposal.get('start')
    if (not isinstance(old,str) or not isinstance(new,str) or not old or not new.strip() or
        max(len(old),len(new))>4000 or old==new or type(start) is not int or start<0 or
        original[start:start+len(old)]!=old or original.count(old)!=1 or
        re.search(r'(?m)^\s*#{2,}',new)):
        return []
    if (re.fullmatch(r'(?:仍需|需要|需|请|建议|待).{0,40}(?:人工|确认|审核|核对).{0,40}',new.strip()) or
        len(old)>6 and difflib.SequenceMatcher(None,old,new,autojunk=False).ratio()<.25 or
        qa.SECTION_RE.search(new)):
        return []
    ranges=editable_ranges(issue,original)
    changes=[op for op in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes() if op[0]!='equal']
    if not changes or any(not any(a<=start+x<=start+y<=b for a,b in ranges)
                          for _,x,y,_,_ in changes):
        return []
    normalized=lambda value:TAG.sub(lambda m:f'<{m[1]}{m[2]}>',value)
    tags=lambda value:set(m[0] for m in TAG.finditer(normalized(value)))
    if tags(new)-tags(old)-set(proposal.get('known_reference_tags',[])):
        return []
    # References may be re-linked to a defined tag, but regular numeric facts
    # and explicit movement direction must not change as a side-effect.
    without_tags=lambda value:TAG.sub('',normalized(value))
    numbers=lambda value:re.findall(r'\d+(?:[.:：/-]\d+)*',without_tags(value))
    if numbers(old)!=numbers(new) or Counter(qa.TIMESTAMP_RE.findall(old))!=Counter(qa.TIMESTAMP_RE.findall(new)):
        return []
    if Counter(MOTION.findall(old))!=Counter(MOTION.findall(new)):
        return []
    for m in TECH.finditer(old):
        if new.count(m[0])!=old.count(m[0]):
            return []
    # Protect audio and screen words. Punctuation-only repairs remain human
    # candidates; they are never default automatic changes inside audio.
    protected=[m.span() for m in SCREEN.finditer(original)]+audio_spans(original)
    result=[]
    for _,a,b,c,d in changes:
        x,y=start+a,start+b
        audio=False
        for p,q in protected:
            overlaps=(x<q and y>p) if x!=y else p<x<q
            if overlaps:
                if _lexical(old[a:b])!=_lexical(new[c:d]):
                    return []
                audio=True
        op={'start':x,'end':y,'old':old[a:b],'new':new[c:d],
            'candidate_old':old,'candidate_new':new,'candidate_policy':POLICY,'review_only':True}
        if audio:op['candidate_scope']='音频或画面文字标点'
        result.append(op)
    return result
