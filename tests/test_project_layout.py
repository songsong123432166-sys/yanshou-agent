"""The unified entry works from other directories without moving local history."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import agent
from captionqa import caption_qa_launcher as launcher, caption_qa_web as web
from tests.test_caption_qa_upgrades import GOOD

ROOT=Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_only_unified_python_entry_remains_at_top_level(self):
        self.assertEqual({p.name for p in ROOT.glob('*.py')},{'agent.py','环境自检.py'} if (ROOT/'环境自检.py').exists() else {'agent.py'})
        self.assertEqual(web.ROOT,ROOT)
        self.assertEqual(launcher.ROOT,ROOT)

    def test_unified_dispatch_passes_arguments_without_opening_a_service(self):
        for mode,module in [('launch','caption_qa_launcher'),('serve','caption_qa_web'),('mimo','run_mimo')]:
            with self.subTest(mode=mode),patch(f'captionqa.{module}.main',return_value=0) as run:
                self.assertEqual(agent.main([mode,'--help']),0)
                run.assert_called_once_with(['--help'])

    def test_absolute_entry_checks_caption_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'result.json'
            result=subprocess.run([sys.executable,str(ROOT/'agent.py'),'check','--dry-run','--text',GOOD,'--output',str(report)],
                cwd=folder,text=True,capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(report.read_text())['semantic_qa_performed'])
            for mode in ('serve','mimo'):
                result=subprocess.run([sys.executable,str(ROOT/'agent.py'),mode,'--help'],cwd=folder,
                    text=True,capture_output=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
