"""CLI acceptance checks without network or credentials."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1]/'agent.py'
CAPTION = "## 主体\n<主体1>：身穿白袍。\n## 总结\n<主体1>奔跑。\n## 自然指令\n<主体1>向前奔跑。\n"


class CLITests(unittest.TestCase):
    def invoke(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), 'check', *args],
                              capture_output=True, encoding="utf-8", timeout=30)

    def test_cli_rejects_nonpositive_concurrency_and_excessive_retries(self) -> None:
        for args in (("--concurrency", "0"), ("--concurrency", "-1"), ("--max-retries", "4")):
            with self.subTest(args=args):
                result = self.invoke("--dry-run", *args)
                self.assertEqual(result.returncode, 2)

    def test_cli_batch_auto_resumes_without_duplicate_results(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"input.jsonl", Path(folder)/"output.jsonl"
            source.write_text("\n".join(json.dumps({"id": ident, "caption": CAPTION})
                                        for ident in ("one", "two")) + "\n", encoding="utf-8")
            args = ("--dry-run", "--input", str(source), "--output", str(target), "--concurrency", "2")
            first, second = self.invoke(*args), self.invoke(*args)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("新增处理 2 条", first.stdout)
            self.assertIn("新增处理 0 条", second.stdout)
            rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["id"] for row in rows}, {"one", "two"})
            self.assertEqual(len(rows), 2)

    def test_cli_subject_renumbering_warning_survives_single_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"report.json"
            caption = CAPTION.replace("## 总结", "<主体1>：穿红衣。\n## 总结")
            result = self.invoke("--dry-run", "--text", caption, "--output", str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NEED_HUMAN_REVIEW")
            self.assertTrue(any(i["issue_type"] == "duplicate_defined_subjects" for i in report["static_errors"]))

    def test_default_safe_auto_fix_and_explicit_review_all(self):
        raw=CAPTION.replace('身穿白袍。','身穿白袍')
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.json'
            result=self.invoke('--dry-run','--text',raw,'--output',str(target))
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(target.read_text())
            self.assertNotEqual(report['revised_caption'],raw)
            self.assertEqual(report['workflow_version'],'safe-auto-review-v3')
            self.assertEqual(report['status'],'AUTO_FIXED')
            manual=self.invoke('--dry-run','--review-all','--text',raw,'--output',str(target))
            self.assertEqual(manual.returncode,0,manual.stderr)
            self.assertEqual(json.loads(target.read_text())['revised_caption'],raw)
            legacy=self.invoke('--dry-run','--legacy-auto-fix','--text',raw,'--output',str(target))
            self.assertEqual(legacy.returncode,0,legacy.stderr)
            self.assertNotEqual(json.loads(target.read_text())['revised_caption'],raw)


if __name__ == "__main__":
    unittest.main()
