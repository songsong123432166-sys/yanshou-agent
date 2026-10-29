"""Offline acceptance tests; no real LLM requests."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import caption_qa_agent as qa


GOOD = "## 主体\n<主体1>：身穿白袍。\n## 总结\n<主体1>奔跑。\n## 自然指令\n0-4秒，第0秒，<主体1>向前跑。\n"


def reply(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = dict(issues=[], need_human_review=False,
                                  human_review_reason="", corrected_sections={})
    result.update(changes)
    return result


class FakeClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=content, refusal=None))])


def issue(section: str, original: str, auto: bool = True) -> dict[str, Any]:
    return dict(section=section, original_snippet=original, issue_type="错别字与衍字",
                suggestion="最小修复", can_auto_fix=auto)


class AcceptanceTests(unittest.TestCase):
    def test_parser_order_and_exact_roundtrip(self) -> None:
        raw = "\n## 任务类型 \r\n动作参考\r\n\r\n## 总结\r\n示例。\r\n"
        sections = qa.parse_caption_sections(raw)
        self.assertEqual(list(sections), ["任务类型", "总结"])
        self.assertEqual(qa.rebuild_caption(raw, sections), raw)

    def test_duplicate_sections_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qa.parse_caption_sections("## 主体\na\n## 主体\nb\n")

    def test_invalid_markdown_preserved_for_review(self) -> None:
        report = qa.CaptionQAAgent(dry_run=True).check("无标题的原文")
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], "无标题的原文")

    def test_subject_sets_numbering_and_exemptions(self) -> None:
        sections = qa.parse_caption_sections("## 主体\n<主体1>。\n<主体3>。\n## 总结\n<主体1><主体4>。\n## 场景\n<场景1>。\n## 画面元素\n<画面元素1>。\n")
        errors = qa.run_static_rules(sections)
        self.assertEqual({e["issue_type"] for e in errors},
                         {"missing_subjects", "ghost_subjects", "skipped_numbering"})

    def test_valid_multidigit_tags_are_not_bare(self) -> None:
        sections = {"提示": "<主体12> <视频10> <图片23> <场景99> 主体12和视频10"}
        errors = qa.run_static_rules(sections)
        self.assertEqual([e["original_snippet"] for e in errors if e["issue_type"] == "bare_tags"], ["主体12", "视频10"])
        self.assertEqual(sum(e["issue_type"] == "missing_punctuation" for e in errors), 1)

    def test_duplicate_prefix_and_punctuation_are_repaired(self) -> None:
        text = GOOD + "## 场景\n<场景1>：<场景1>：<场景1>：山洞\n\n## 输入描述\n<图片1>：黑板\n<视频1>：蜡烛\n## 画面元素\n无\n"
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("<场景1>：<场景1>：<场景1>：山洞。", report["revised_caption"])
        self.assertIn("<图片1>：黑板。\n<视频1>：蜡烛。", report["revised_caption"])
        self.assertEqual(report["remaining_static_errors"], [])
        self.assertFalse(report["semantic_qa_performed"])

    def test_pass_requires_clean_checks(self) -> None:
        report = qa.CaptionQAAgent(client=FakeClient(reply())).check(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["semantic_qa_performed"])
        self.assertEqual(report["revised_caption"], GOOD)

    def test_minimal_llm_fix(self) -> None:
        text = GOOD.replace("身穿白袍", "身穿白袍，露出眼晴")
        client = FakeClient(reply(issues=[issue("主体", "眼晴")],
            corrected_sections={"主体": "<主体1>：身穿白袍，露出眼睛。"}))
        report = qa.CaptionQAAgent(client=client).check(text)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("露出眼睛", report["revised_caption"])
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0]["messages"][0]["content"], qa.LLM_SYSTEM_PROMPT)

    def test_implicit_label_insertion_requires_review(self) -> None:
        text = GOOD.replace("<主体1>奔跑。", "那名女子奔跑。")
        problem = issue("总结", "那名女子")
        problem["issue_type"] = "隐式漏挂主体标签"
        client = FakeClient(reply(issues=[problem], corrected_sections={"总结": "<主体1>那名女子奔跑。"}))
        report = qa.CaptionQAAgent(client=client).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], text)
        self.assertFalse(report["issues"][0]["can_auto_fix"])
        self.assertFalse(report["issues"][0]["resolved"])

    def test_new_non_subject_entity_rejected(self) -> None:
        client = FakeClient(reply(issues=[issue("总结", "<主体1>奔跑。")],
            corrected_sections={"总结": "<主体1>在<场景99>奔跑。"}))
        report = qa.CaptionQAAgent(client=client).check(GOOD)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], GOOD)

    def test_missing_subject_cannot_be_hallucinated_away(self) -> None:
        text = GOOD.replace("## 总结", "<主体2>：红衣。\n## 总结")
        client = FakeClient(reply(issues=[issue("自然指令", "<主体1>向前跑")],
            corrected_sections={"自然指令": "0-4秒，第0秒，<主体1>向前跑，<主体2>倒地。"}))
        report = qa.CaptionQAAgent(client=client).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], text)

    def test_timestamp_change_rejected(self) -> None:
        client = FakeClient(reply(issues=[issue("自然指令", "0-4秒")],
            corrected_sections={"自然指令": "1-5秒，第0秒，<主体1>向前跑。"}))
        report = qa.CaptionQAAgent(client=client).check(GOOD)
        self.assertEqual(report["revised_caption"], GOOD)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")

    def test_unreported_rewrite_is_rejected(self) -> None:
        client = FakeClient(reply(corrected_sections={"总结": "<主体1>在阳光下优雅地奔跑。"}))
        report = qa.CaptionQAAgent(client=client).check(GOOD)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], GOOD)

    def test_manual_issue_is_never_silently_applied(self) -> None:
        client = FakeClient(reply(issues=[issue("主体", "白袍", False)],
            corrected_sections={"主体": "<主体1>：身穿红袍。"}))
        report = qa.CaptionQAAgent(client=client).check(GOOD)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], GOOD)

    def test_retry_invalid_json_then_success(self) -> None:
        client = FakeClient("{broken", reply())
        report = qa.CaptionQAAgent(client=client, retry_delay=0).check(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(client.calls), 2)

    def test_schema_failure_falls_back_without_false_pass(self) -> None:
        client = FakeClient({"issues": [], "need_human_review": "false"})
        report = qa.CaptionQAAgent(client=client, max_retries=0).check(GOOD)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertFalse(report["semantic_qa_performed"])
        self.assertEqual(report["revised_caption"], GOOD)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "optional SDK not installed")
    def test_official_sdk_chat_completions_wire_format(self) -> None:
        import httpx
        from openai import OpenAI
        requests: list[dict[str, Any]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "offline-test", "object": "chat.completion",
                "created": 0, "model": "offline-model", "choices": [{"index": 0,
                "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(reply())}}]})

        with OpenAI(api_key="offline-test-key", base_url="https://caption.invalid/v1",
                    max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            report = qa.CaptionQAAgent(client=client, model_name="offline-model").check(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(requests[0]["model"], "offline-model")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})

    def test_jsonl_batch_isolates_bad_records_and_preserves_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"in.jsonl", Path(directory)/"out.csv"
            source.write_text(json.dumps({"id": 1, "caption": GOOD}) + '\n{bad\n' +
                              json.dumps({"id": 2, "caption": GOOD}), encoding="utf-8")
            count = qa.process_batch(source, target, qa.CaptionQAAgent(dry_run=True))
            self.assertEqual(count, 3)
            with target.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["id"], "1")
            self.assertEqual(rows[1]["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")
            self.assertEqual(rows[2]["id"], "2")
            self.assertEqual(rows[0]["qa_revised_caption"], "")

    def test_json_and_csv_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for ext in ("json", "csv"):
                source, target = Path(directory)/f"in.{ext}", Path(directory)/f"out-{ext}.jsonl"
                if ext == "json":
                    source.write_text(json.dumps([{"caption": GOOD, "id": "中文"}]), encoding="utf-8")
                else:
                    with source.open("w", encoding="utf-8-sig", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["caption", "id"])
                        writer.writeheader()
                        writer.writerow({"caption": GOOD, "id": "中文"})
                qa.process_batch(source, target, qa.CaptionQAAgent(dry_run=True))
                row = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual(row["id"], "中文")
                self.assertEqual(row["qa_status"], "PASS")

    def test_markdown_report_has_requested_blocks(self) -> None:
        out = io.StringIO()
        qa.print_markdown_report(qa.CaptionQAAgent(dry_run=True).check(GOOD), stream=out)
        for title in ("1. 主体与标签核查结果", "2. 问题清单表格", "3. 订正后完整 Caption"):
            self.assertIn(title, out.getvalue())


if __name__ == "__main__":
    unittest.main()
