"""Review contract and resolved values, independent of HTTP and file formats."""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

WORKFLOW = 'review-first-v1'
PREVIOUS_AUTO_WORKFLOW = 'safe-auto-review-v2'
AUTO_WORKFLOW = 'safe-auto-review-v3'
AUTO_WORKFLOWS = {PREVIOUS_AUTO_WORKFLOW, AUTO_WORKFLOW}
DECISION_LABELS = {
    'pending':'待审核', 'approve':'确认按建议改', 'auto':'已自动修正',
    'reject':'误报', 'keep':'保留原文', 'keep_highlight':'保留原文并标黄',
    'defer':'暂无法判断', 'manual':'已人工修改',
}
DECISIONS = set(DECISION_LABELS)
PROJECTION_VERSION = 1


def freeze(value):
    """Snapshot nested JSON metadata without sharing mutable lists or maps."""
    if isinstance(value, Mapping):
        return MappingProxyType({key:freeze(item) for key,item in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {key:thaw(item) for key,item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FieldResolution:
    field_id: str
    row_index: int
    sample_id: str
    name: str
    original: str
    final_text: str
    editor_text: str
    mode: str
    error: str = ''

    @property
    def changed(self):
        return not self.error and self.final_text != self.original

    def public(self):
        return {'final_text':self.final_text, 'editor_text':self.editor_text,
                'mode':self.mode, 'changed':self.changed, 'error':self.error}


@dataclass(frozen=True)
class CaptionResolution:
    row_index: int
    sample_id: str
    original: str
    final_text: str
    field_ids: tuple[str, ...]
    highlighted: bool
    errors: tuple[str, ...] = ()

    @property
    def changed(self):
        return not self.errors and self.final_text != self.original
