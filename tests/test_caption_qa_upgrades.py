"""Offline regressions for sliced semantic QA and asynchronous single-record calls."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import captionqa.caption_qa_agent as qa


GOOD = (
    "## 主体\n<主体1>：身穿白袍。\n"
    "## 总结\n<主体1>奔跑。\n"
    "## 自然指令\n0-4秒，第0秒，<主体1>向前跑。\n"
)
TARGET_SECTIONS = {"提示", "输入描述", "基调", "场景", "主体", "总结", "自然指令"}


def semantic_reply(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "issues": [], "need_human_review": False,
        "human_review_reason": "", "corrected_sections": {},
    }
    result.update(changes)
    return result


def semantic_issue(section: str, original: str, suggestion: str,
                   kind: str = "错别字与衍字") -> dict[str, Any]:
    return {
        "section": section, "original_snippet": original,
        "issue_type": kind, "suggestion": suggestion, "can_auto_fix": True,
    }


def sdk_response(value: Any) -> Any:
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])


class FakeSyncClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return sdk_response(value)


class FakeAsyncClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self.close_calls = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return sdk_response(value)

    async def close(self) -> None:
        self.close_calls += 1


class FakeRateLimitError(Exception):
    status_code = 429


class FakeAuthenticationError(Exception):
    status_code = 401


def request_sections(call: dict[str, Any]) -> dict[str, str]:
    payload = json.loads(call["messages"][1]["content"])
    return payload["caption_sections"]


class StaticAndMergeUpgradeTests(unittest.TestCase):
    def test_missing_subject_still_allows_existing_text_typo_fix(self) -> None:
        text = GOOD.replace("## 总结", "<主体2>：红衣。\n## 总结").replace(
            "<主体1>奔跑。", "<主体1>睁开眼晴奔跑。")
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("总结", "眼晴", "眼睛")],
            corrected_sections={"总结": "<主体1>睁开眼睛奔跑。"}))
        report = qa.CaptionQAAgent(client=client).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertIn("<主体1>睁开眼睛奔跑。", report["revised_caption"])
        self.assertNotIn("眼晴", report["revised_caption"])
        self.assertEqual(report["subject_check"]["missing_subjects"], ["<主体2>"])

    def test_missing_subject_blocks_unlabelled_new_action_insertion(self) -> None:
        text = GOOD.replace("## 总结", "<主体2>：红衣。\n## 总结")
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("总结", "<主体1>奔跑。", "补全语句", "语法病句与语句不通顺")],
            corrected_sections={"总结": "<主体1>奔跑，红衣人倒地。"}))
        report = qa.CaptionQAAgent(client=client).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], text)

    def test_duplicate_subject_definitions_have_exact_warning_and_force_review(self) -> None:
        text = GOOD.replace(
            "## 总结", "<主体1>：身穿红袍。\n## 总结")
        errors = qa.run_static_rules(qa.parse_caption_sections(text))
        duplicates = [item for item in errors
                      if item["issue_type"] == "duplicate_defined_subjects"]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["original_snippet"], "<主体1>")
        self.assertEqual(duplicates[0]["suggestion"],
            "【主体编号重复定义】在 ## 主体 中重复定义了 <主体1>，请核对是否应顺延编号")
        report = qa.CaptionQAAgent(dry_run=True).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertTrue(report["need_human_review"])

    def test_duplicate_definition_cannot_be_cleared_by_llm_deletion(self) -> None:
        text = GOOD.replace(
            "## 总结", "<主体1>：身穿红袍。\n## 总结")
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("主体", "<主体1>：身穿红袍。", "核对主体编号")],
            corrected_sections={"主体": "<主体1>：身穿白袍。"}))
        report = qa.CaptionQAAgent(client=client, retry_delay=0).check(text)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        duplicate = next(item for item in report["static_errors"]
                         if item["issue_type"] == "duplicate_defined_subjects")
        self.assertFalse(duplicate["resolved"])

    def test_different_subject_numbers_are_not_duplicate_definitions(self) -> None:
        text = GOOD.replace("## 总结", "<主体2>：身穿红袍。\n## 总结")
        errors = qa.run_static_rules(qa.parse_caption_sections(text))
        self.assertNotIn("duplicate_defined_subjects", {item["issue_type"] for item in errors})

    def test_system_prompt_contains_objective_grammar_and_differential_contract(self) -> None:
        for text in ("句式杂糅", "语序颠倒", "中途暗换主语", "只返回实际发生了",
                     "语法病句与语句不通顺", "严禁挑刺", "时间戳逻辑"):
            with self.subTest(text=text):
                self.assertIn(text, qa.LLM_SYSTEM_PROMPT)

    def test_llm_receives_all_and_only_whitelisted_sections(self) -> None:
        raw = (
            "## 任务类型 \n人物参考|造型参考\n"
            "## similarity_label\n yes  \n"
            "## video_similarity_tag\n实体/IP - 相同\n"
            "## audio_similarity_tag\n无声音\n"
            "## 提示\n参考<视频1>生成视频。\n"
            "## 输入描述\n<视频1>：女子奔跑。\n"
            "## 基调\n写实。\n"
            "## 场景\n<场景1>：山洞。\n" + GOOD +
            "## 画面元素\n<画面元素1>：灯。\n"
        )
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(raw)
        self.assertEqual(set(request_sections(client.calls[0])), TARGET_SECTIONS|{'画面元素'})
        self.assertEqual(json.loads(client.calls[0]['messages'][1]['content'])['read_only_sections'],['画面元素'])
        self.assertEqual(report["revised_caption"], raw)
        self.assertEqual(set(qa.LLM_TARGET_SECTIONS), TARGET_SECTIONS)

    def test_llm_receives_deterministically_fixed_sections_before_semantics(self) -> None:
        raw = GOOD + "## 场景\n<场景1>：<场景1>：石质台阶\n"
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(raw)
        self.assertEqual(request_sections(client.calls[0])["场景"].strip(),
                         "<场景1>：<场景1>：石质台阶。")
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("<场景1>：<场景1>：石质台阶。", report["revised_caption"])

    def test_exempt_metadata_is_preserved_and_frame_elements_only_gain_period(self) -> None:
        metadata = (
            "## 任务类型 \r\n  视频1 | 主体7  \r\n\r\n"
            "## similarity_label \r\n yes\t \r\n"
            "## video_similarity_tag\r\n视频1|场景3 \r\n"
            "## audio_similarity_tag\r\n无声音\r\n"
            "## 画面元素\r\n主体3和图片2；<场景1>：<场景1>：原样  \r\n"
        )
        raw = metadata + GOOD
        client = FakeSyncClient(semantic_reply())
        report = qa.CaptionQAAgent(client=client).check(raw)
        expected = raw.replace("原样  \r\n", "原样。  \r\n")
        self.assertEqual(report["revised_caption"], expected)
        self.assertTrue(set(request_sections(client.calls[0])).issubset(TARGET_SECTIONS|{'画面元素'}))
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual([i["issue_type"] for i in report["issues"]], ["missing_punctuation"])

    def test_metadata_format_errors_are_not_leaked_through_request_errors(self) -> None:
        raw = "## 画面元素\n<画面元素1>：右上角【台标】。\n## similarity_label\nPRIVATE_META\n" + GOOD
        client = FakeSyncClient(semantic_reply())
        qa.CaptionQAAgent(client=client).check(raw)
        payload = client.calls[0]["messages"][1]["content"]
        self.assertIn('画面元素1',payload)
        self.assertNotIn('PRIVATE_META',payload)

    def test_llm_cannot_modify_metadata_via_unsolicited_response(self) -> None:
        raw = "## similarity_label\n yes \n" + GOOD
        client = FakeSyncClient(semantic_reply(corrected_sections={"similarity_label": "no"}))
        report = qa.CaptionQAAgent(client=client, max_retries=0).check(raw)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertEqual(report["revised_caption"], raw)

    def test_partial_corrected_sections_preserve_untouched_fields_and_order(self) -> None:
        raw = (
            "## similarity_label \n yes \n\n"
            "## 输入描述\n<视频1>：眼晴特写。\n" + GOOD +
            "## 画面元素\n 无 \n"
        )
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("输入描述", "眼晴", "眼睛")],
            corrected_sections={"输入描述": "<视频1>：眼睛特写。"}))
        report = qa.CaptionQAAgent(client=client).check(raw)
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertEqual(report["revised_caption"], raw.replace("眼晴", "眼睛").replace(" 无 \n", " 无。 \n"))
        self.assertEqual(list(qa.parse_caption_sections(report["revised_caption"])),
                         list(qa.parse_caption_sections(raw)))

    def test_grammar_issue_type_is_accepted_but_not_auto_applied(self) -> None:
        raw = GOOD.replace("<主体1>奔跑。", "<主体1>依次做出、痛苦挣扎的动作。")
        client = FakeSyncClient(semantic_reply(
            issues=[semantic_issue("总结", "依次做出、痛苦挣扎的动作", "删除多余顿号",
                                   "语法病句与语句不通顺")],
            corrected_sections={"总结": "<主体1>依次做出痛苦挣扎的动作。"}))
        report = qa.CaptionQAAgent(client=client, retry_delay=0, max_retries=0).check(raw)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertTrue(report["semantic_qa_performed"])
        self.assertEqual(report["revised_caption"], raw)


class AsyncSingleCaptionUpgradeTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(importlib.util.find_spec("openai"), "optional SDK not installed")
    async def test_official_async_sdk_chat_completions_wire_format(self) -> None:
        import httpx
        from openai import AsyncOpenAI

        requests: list[dict[str, Any]] = []

        async def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.method, "POST")
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "offline-async-test", "object": "chat.completion", "created": 0,
                "model": "offline-async-model", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps(semantic_reply())}}],
            })

        async with AsyncOpenAI(
            api_key="offline-test-key", base_url="https://caption.invalid/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            async with qa.CaptionQAAgent(async_client=client, model_name="offline-async-model") as agent:
                report = await agent.acheck_single_caption(GOOD)
            self.assertFalse(client.is_closed())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["model"], "offline-async-model")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})

    async def test_injected_client_lifetime_remains_owned_by_caller(self) -> None:
        client = FakeAsyncClient(semantic_reply(), semantic_reply())
        async with qa.CaptionQAAgent(async_client=client) as agent:
            first = await agent.acheck_single_caption(GOOD)
        await agent.aclose()
        second = await agent.acheck_single_caption(GOOD)
        self.assertEqual((first["status"], second["status"]), ("PASS", "PASS"))
        self.assertEqual(client.close_calls, 0)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "optional SDK not installed")
    async def test_owned_async_client_is_shared_and_closed_once_at_context_exit(self) -> None:
        client = FakeAsyncClient(semantic_reply(), semantic_reply())
        with patch("openai.AsyncOpenAI", return_value=client) as factory:
            async with qa.CaptionQAAgent(api_key="offline-test-key") as agent:
                reports = await asyncio.gather(agent.acheck_single_caption(GOOD),
                                               agent.acheck_single_caption(GOOD))
            await agent.aclose()
        self.assertTrue(all(report["status"] == "PASS" for report in reports))
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.close_calls, 1)

    async def test_async_single_caption_uses_json_object_chat_completions(self) -> None:
        client = FakeAsyncClient(semantic_reply())
        agent = qa.CaptionQAAgent(async_client=client, retry_delay=0)
        report = await agent.acheck_single_caption(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["semantic_qa_performed"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0]["messages"][0]["content"], qa.LLM_SYSTEM_PROMPT)

    async def test_async_retry_malformed_json_three_times_then_success(self) -> None:
        client = FakeAsyncClient("{bad", "{bad", "{bad", semantic_reply())
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(client.calls), 4)

    async def test_async_retry_rate_limit_three_times_then_success(self) -> None:
        client = FakeAsyncClient(FakeRateLimitError(), FakeRateLimitError(),
                                 FakeRateLimitError(), semantic_reply())
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(client.calls), 4)

    async def test_async_retry_timeout_three_times_then_success(self) -> None:
        client = FakeAsyncClient(TimeoutError(), TimeoutError(), TimeoutError(), semantic_reply())
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(GOOD)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(client.calls), 4)

    async def test_async_retry_exhaustion_keeps_caption_for_review(self) -> None:
        client = FakeAsyncClient("{bad", "{bad", "{bad", "{bad")
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(GOOD)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
        self.assertFalse(report["semantic_qa_performed"])
        self.assertEqual(report["revised_caption"], GOOD)

    async def test_async_authentication_error_is_not_retried(self) -> None:
        client = FakeAsyncClient(FakeAuthenticationError())
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(GOOD)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")

    async def test_concurrent_single_calls_never_mix_caption_contexts(self) -> None:
        client = FakeAsyncClient(*(semantic_reply() for _ in range(3)))
        agent = qa.CaptionQAAgent(async_client=client, retry_delay=0)
        captions = [GOOD.replace("身穿白袍", f"身穿白袍，唯一标记甲{index}")
                    for index in range(3)]
        reports = await asyncio.gather(*(agent.acheck_single_caption(text) for text in captions))
        self.assertTrue(all(report["status"] == "PASS" for report in reports))
        self.assertEqual(len(client.calls), 3)
        found: set[int] = set()
        for call in client.calls:
            payload_text = json.dumps(request_sections(call), ensure_ascii=False)
            matches = [index for index in range(3) if f"唯一标记甲{index}" in payload_text]
            self.assertEqual(len(matches), 1)
            found.update(matches)
        self.assertEqual(found, {0, 1, 2})

    async def test_async_and_sync_share_static_first_partial_merge_behavior(self) -> None:
        raw = GOOD.replace("身穿白袍", "身穿白袍，露出眼晴") + "## 场景\n<场景1>：<场景1>：山洞\n"
        client = FakeAsyncClient(semantic_reply(
            issues=[semantic_issue("主体", "眼晴", "眼睛")],
            corrected_sections={"主体": "<主体1>：身穿白袍，露出眼睛。"}))
        report = await qa.CaptionQAAgent(async_client=client, retry_delay=0).acheck_single_caption(raw)
        self.assertEqual(request_sections(client.calls[0])["场景"].strip(), "<场景1>：<场景1>：山洞。")
        self.assertEqual(report["status"], "AUTO_FIXED")
        self.assertIn("<主体1>：身穿白袍，露出眼睛。", report["revised_caption"])
        self.assertIn("<场景1>：<场景1>：山洞。", report["revised_caption"])


if __name__ == "__main__":
    unittest.main()
