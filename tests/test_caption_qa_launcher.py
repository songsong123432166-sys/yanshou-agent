"""Launcher reuses live jobs and never terminates a service on conflicts."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

import captionqa.caption_qa_launcher as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'agent.py').write_text('# synthetic service')

    def test_running_agent_is_reused_without_spawn_or_credential_access(self):
        with patch.object(launcher,'probe',return_value='ready'),patch.object(launcher,'open_page') as opened,patch.object(launcher.subprocess,'Popen') as spawn:
            message=launcher.launch(root=self.root)
        self.assertIn('正在运行',message);opened.assert_called_once_with('http://127.0.0.1:8765');spawn.assert_not_called()

    def test_conflicting_service_is_not_started_or_stopped(self):
        with patch.object(launcher,'probe',return_value='occupied'),patch.object(launcher.subprocess,'Popen') as spawn,patch.object(launcher,'open_page') as opened:
            with self.assertRaisesRegex(launcher.LaunchError,'未停止任何进程'):launcher.launch(root=self.root)
        spawn.assert_not_called();opened.assert_not_called()

    def test_new_server_is_detached_and_browser_opens_only_when_ready(self):
        child=Mock(pid=12345);child.poll.return_value=None
        with patch.object(launcher,'probe',side_effect=['absent','occupied','ready']),patch.object(launcher.subprocess,'Popen',return_value=child) as spawn,patch.object(launcher,'open_page') as opened,patch.object(launcher.time,'sleep'):
            launcher.launch(root=self.root)
        self.assertTrue(spawn.call_args.kwargs['start_new_session']);self.assertEqual(spawn.call_args.kwargs['stdin'],subprocess.DEVNULL)
        self.assertNotIn('--open',spawn.call_args.args[0]);opened.assert_called_once()
        self.assertEqual((self.root/'.local/launcher/server.pid').read_text(),'12345\n')

    def test_child_exit_and_timeout_do_not_claim_success_or_kill_jobs(self):
        child=Mock();child.poll.return_value=1
        with patch.object(launcher,'probe',side_effect=['absent','occupied']),patch.object(launcher.subprocess,'Popen',return_value=child),patch.object(launcher,'open_page') as opened:
            with self.assertRaisesRegex(launcher.LaunchError,'启动未完成'):launcher.launch(root=self.root)
        child.terminate.assert_not_called();child.kill.assert_not_called();opened.assert_not_called()
        with patch.object(launcher,'probe',return_value='absent'),patch.object(launcher.subprocess,'Popen',return_value=child):
            with self.assertRaisesRegex(launcher.LaunchError,'不会终止后台任务'):launcher.launch(root=self.root,timeout=0)
        child.kill.assert_not_called()

    def test_verification_does_not_open_browser(self):
        with patch.object(launcher,'probe',return_value='ready'),patch.object(launcher,'open_page') as opened:
            launcher.launch(root=self.root,open_browser=False)
        opened.assert_not_called()


if __name__=='__main__':unittest.main()
