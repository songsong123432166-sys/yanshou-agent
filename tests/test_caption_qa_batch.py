"""Offline async batch/checkpoint acceptance tests; never sends API requests."""
from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import captionqa.caption_qa_agent as qa


GOOD = "## 主体\n<主体1>：白袍。\n## 总结\n<主体1>奔跑。\n## 自然指令\n<主体1>向前奔跑。\n"


class FakeAsyncAgent:
    """Network boundary substitute; actual report/record assembly remains exercised."""

    def __init__(self, *, dry_run: bool = False, fail_semantic: bool = False) -> None:
        self.dry_run = dry_run
        self.fail_semantic = fail_semantic
        self.llm_options = {"model_name": "offline-model", "base_url": "https://caption.invalid/v1"}
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def acheck_single_caption(self, caption: str) -> qa.QAReport:
        self.calls.append(caption)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            report = qa.CaptionQAAgent(dry_run=True).check(caption)
            report["qa_scope"] = "static_only" if self.dry_run else "static_and_semantic"
            report["semantic_qa_performed"] = not self.dry_run and not self.fail_semantic
            if self.fail_semantic:
                report.update(status="NEED_HUMAN_REVIEW", need_human_review=True,
                              human_review_reason="offline transient timeout")
            return report
        finally:
            self.active -= 1

    @staticmethod
    def failure_report(caption: str, reason: str) -> qa.QAReport:
        return qa.CaptionQAAgent.failure_report(caption, reason)

    async def aclose(self) -> None:
        self.closed = True


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AsyncBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "input.jsonl"
        self.target = Path(self.directory.name) / "output.jsonl"

    async def test_async_batch_entry_point_exists(self) -> None:
        self.assertTrue(callable(getattr(qa, "aprocess_batch", None)), "missing asynchronous checkpoint batch entry point")

    async def test_concurrency_is_bounded_and_every_caption_is_isolated(self) -> None:
        records = [{"id": i, "caption": GOOD + f"## 提示\n样本{i}。\n"} for i in range(25)]
        write_jsonl(self.source, records)
        agent = FakeAsyncAgent()
        count = await qa.aprocess_batch(self.source, self.target, agent, concurrency=3)
        rows = read_rows(self.target)
        self.assertEqual(count, 25)
        self.assertEqual(set(agent.calls), {row["caption"] for row in records})
        self.assertEqual(agent.max_active, 3)
        self.assertTrue(agent.closed)
        self.assertEqual({row["id"] for row in rows}, set(range(25)))
        self.assertTrue(all(row["qa_checkpoint"]["completed"] for row in rows))

    async def test_finished_rows_are_flushed_before_slow_request_returns(self) -> None:
        slow = GOOD + "## 提示\nslow\n"
        write_jsonl(self.source, [{"id": "slow", "caption": slow}, {"id": "fast", "caption": GOOD}])
        release, fast_returned = asyncio.Event(), asyncio.Event()
        agent = FakeAsyncAgent()
        original = agent.acheck_single_caption

        async def blocked(caption: str) -> qa.QAReport:
            if caption == slow:
                await release.wait()
            report = await original(caption)
            if caption != slow:
                fast_returned.set()
            return report

        agent.acheck_single_caption = blocked  # type: ignore[method-assign]
        task = asyncio.create_task(qa.aprocess_batch(self.source, self.target, agent, concurrency=2))
        try:
            await asyncio.wait_for(fast_returned.wait(), timeout=2)
            for _ in range(100):
                if self.target.exists() and self.target.stat().st_size:
                    break
                await asyncio.sleep(0.001)
            self.assertFalse(task.done())
            self.assertEqual([row["id"] for row in read_rows(self.target)], ["fast"])
        finally:
            release.set()
            await task

    async def test_resuming_with_eight_workers_only_checks_unfinished_captions(self) -> None:
        records = [{"id": i, "caption": GOOD + f"## 提示\n样本{i}。\n"} for i in range(24)]
        write_jsonl(self.source, records)
        finished = {row["caption"] for row in records[:8]}
        first = FakeAsyncAgent()
        original_check = first.acheck_single_caption

        async def partial_check(caption):
            report = await original_check(caption)
            if caption not in finished:
                report["semantic_qa_performed"] = False
            return report

        first.acheck_single_caption = partial_check
        await qa.aprocess_batch(self.source, self.target, first, concurrency=3)
        original_bytes = self.target.read_bytes()
        resumed = FakeAsyncAgent()
        count = await qa.aprocess_batch(self.source, self.target, resumed, concurrency=8)
        self.assertEqual(count, 16)
        self.assertEqual(resumed.max_active, 8)
        self.assertEqual(set(resumed.calls), {row["caption"] for row in records[8:]})
        self.assertTrue(self.target.read_bytes().startswith(original_bytes))
        latest = {row["row_index"]: row for row in read_rows(self.target)}
        self.assertEqual(len(latest), 24)
        self.assertTrue(all(row["qa_checkpoint"]["completed"] for row in latest.values()))

    async def test_resume_distinguishes_typed_ids_and_uses_physical_line_numbers(self) -> None:
        records = [{"id": 1, "caption": GOOD}, {"id": "1", "caption": GOOD},
                   {"id": None, "caption": GOOD}, {"id": "", "caption": GOOD}, {"caption": GOOD}]
        self.source.write_text("\n" + "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        agent = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent), 6)
        rows = read_rows(self.target)
        self.assertEqual(len({row["qa_checkpoint"]["key"] for row in rows}), 6)
        self.assertEqual(sorted(row["row_index"] for row in rows), [1, 2, 3, 4, 5, 6])
        self.assertEqual(next(row for row in rows if row["row_index"] == 1)["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")
        resumed = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, resumed), 0)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(len(read_rows(self.target)), 6)

    async def test_semantic_transport_failure_is_written_but_retried_on_resume(self) -> None:
        write_jsonl(self.source, [{"id": "retry", "caption": GOOD}])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(fail_semantic=True))
        self.assertFalse(read_rows(self.target)[0]["qa_checkpoint"]["completed"])
        resumed = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, resumed), 1)
        self.assertEqual(len(resumed.calls), 1)
        self.assertTrue(read_rows(self.target)[1]["qa_checkpoint"]["completed"])
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent()), 0)

    async def test_malformed_input_is_local_review_and_completed(self) -> None:
        self.source.write_text(json.dumps({"id": "ok", "caption": GOOD}) + "\n{broken\n", encoding="utf-8")
        agent = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent), 2)
        malformed = next(row for row in read_rows(self.target) if row["row_index"] == 2)
        self.assertEqual(malformed["_raw_line"], "{broken")
        self.assertEqual(malformed["qa_status"], "SKIPPED_NOT_HUMAN_CAPTION")
        self.assertTrue(malformed["qa_checkpoint"]["completed"])
        self.assertEqual(agent.calls, [GOOD])

    async def test_changed_input_or_scope_cannot_silently_reuse_checkpoint(self) -> None:
        write_jsonl(self.source, [{"id": "ok", "caption": GOOD}])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(dry_run=True))
        write_jsonl(self.source, [{"id": "ok", "caption": GOOD + "\n"}])
        agent = FakeAsyncAgent()
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, agent)
        self.assertEqual(agent.calls, [])

    async def test_duplicate_explicit_ids_reject_before_any_request_or_output(self) -> None:
        write_jsonl(self.source, [{"id": "same", "caption": GOOD}, {"id": "same", "caption": GOOD}])
        agent = FakeAsyncAgent()
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, agent)
        self.assertEqual(agent.calls, [])
        self.assertFalse(self.target.exists())

    async def test_torn_final_checkpoint_line_is_repaired_but_middle_corruption_rejects(self) -> None:
        write_jsonl(self.source, [{"id": "ok", "caption": GOOD}])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())
        with self.target.open("ab") as stream:
            stream.write(b'{"qa_che')
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent()), 0)
        valid = self.target.read_bytes()
        self.assertEqual(len(read_rows(self.target)), 1)
        self.target.write_bytes(b"{broken\n" + valid)
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())

    async def test_existing_input_columns_are_preserved_in_collision_backup(self) -> None:
        original = {"id": 10, "caption": GOOD, "qa_status": "input status", "qa_checkpoint": {"user": True},
                    "row_index": "user row", "qa_previous_values": "input backup"}
        write_jsonl(self.source, [original])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())
        row = read_rows(self.target)[0]
        self.assertEqual(row["qa_previous_values"], "input backup")
        self.assertEqual(row["qa_previous_values_"]["qa_status"], "input status")
        self.assertEqual(row["qa_previous_values_"]["qa_checkpoint"], {"user": True})
        self.assertEqual(row["qa_previous_values_"]["row_index"], "user row")

    async def test_csv_resume_uses_original_record_index_and_custom_caption_field(self) -> None:
        self.source = self.source.with_suffix(".csv")
        with self.source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "extra"])
            writer.writeheader()
            writer.writerow({"id": "", "text": GOOD, "extra": "中文"})
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(), caption_field="text")
        row = read_rows(self.target)[0]
        self.assertEqual(row["extra"], "中文")
        self.assertEqual(row["row_index"], 1)
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(), caption_field="text"), 0)

    async def test_invalid_concurrency_rejects(self) -> None:
        write_jsonl(self.source, [{"caption": GOOD}])
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(), concurrency=value)

    async def test_final_complete_checkpoint_without_newline_is_preserved_before_append(self) -> None:
        write_jsonl(self.source, [{"id": "done", "caption": GOOD}, {"id": "retry", "caption": GOOD}])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent(fail_semantic=True))
        rows = read_rows(self.target)
        rows[0]["qa_checkpoint"]["completed"] = True
        self.target.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        agent = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, agent), 1)
        self.assertEqual(len(read_rows(self.target)), 3)
        self.assertEqual(agent.calls, [GOOD])

    async def test_cancellation_keeps_flushed_rows_and_cancels_inflight_requests(self) -> None:
        slow = GOOD + "## 提示\nslow\n"
        write_jsonl(self.source, [{"id": "slow", "caption": slow}, {"id": "fast", "caption": GOOD}])
        block, slow_cancelled = asyncio.Event(), asyncio.Event()
        agent = FakeAsyncAgent()
        original = agent.acheck_single_caption

        async def blocked(caption: str) -> qa.QAReport:
            if caption == slow:
                try:
                    await block.wait()
                except asyncio.CancelledError:
                    slow_cancelled.set()
                    raise
            return await original(caption)

        agent.acheck_single_caption = blocked  # type: ignore[method-assign]
        task = asyncio.create_task(qa.aprocess_batch(self.source, self.target, agent, concurrency=2))
        try:
            for _ in range(1000):
                if self.target.exists() and self.target.stat().st_size:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual([row["id"] for row in read_rows(self.target)], ["fast"])
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(slow_cancelled.is_set())
        self.assertTrue(agent.closed)
        resumed = FakeAsyncAgent()
        self.assertEqual(await qa.aprocess_batch(self.source, self.target, resumed), 1)
        self.assertEqual(resumed.calls, [slow])

    async def test_checkpoint_key_tampering_and_legacy_output_are_rejected(self) -> None:
        write_jsonl(self.source, [{"id": "original", "caption": GOOD}])
        await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())
        row = read_rows(self.target)[0]
        row["qa_checkpoint"]["key"] = '["id","str","another"]'
        write_jsonl(self.target, [row])
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())
        write_jsonl(self.target, [{"id": "original", "qa_status": "PASS"}])
        with self.assertRaises(ValueError):
            await qa.aprocess_batch(self.source, self.target, FakeAsyncAgent())


if __name__ == "__main__":
    unittest.main()
