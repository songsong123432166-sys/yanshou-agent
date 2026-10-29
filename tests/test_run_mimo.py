"""Credential storage and launcher checks; no real Keychain or API mutations."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import unittest
from unittest.mock import patch

import captionqa.run_mimo as mimo

DUMMY = "tp-offline-test-key"


class MiMoLauncherTests(unittest.TestCase):
    def test_save_sends_secret_via_stdin_never_command_arguments(self):
        replies = [subprocess.CompletedProcess([], 0, DUMMY.encode(), b""),
                   subprocess.CompletedProcess([], 0, b"", b""),
                   subprocess.CompletedProcess([], 0, DUMMY.encode(), b"")]
        with patch.object(mimo, "_macos"), patch.object(mimo.subprocess, "run", side_effect=replies) as run:
            mimo.save_key_from_clipboard()
        for call in run.call_args_list:
            self.assertNotIn(DUMMY, str(call.args))
            self.assertTrue(call.kwargs["capture_output"])
        write = run.call_args_list[1]
        self.assertEqual(write.args[0], [mimo.SECURITY, "-i"])
        self.assertIn(DUMMY.encode(), write.kwargs["input"])
        self.assertIn(b"-U", write.kwargs["input"])
        read_args = run.call_args_list[2].args[0]
        self.assertIn(mimo.SERVICE, read_args)
        self.assertIn(mimo.ACCOUNT, read_args)

    def test_masked_clipboard_is_rejected_before_any_keychain_write(self):
        reply = subprocess.CompletedProcess([], 0, b"tp-********", b"")
        with patch.object(mimo, "_macos"), patch.object(mimo.subprocess, "run", return_value=reply) as run:
            with self.assertRaises(mimo.CredentialError):
                mimo.save_key_from_clipboard()
        self.assertEqual(run.call_count, 1)

    def test_mismatched_roundtrip_is_reported_without_secret(self):
        replies = [subprocess.CompletedProcess([], 0, DUMMY.encode(), b""),
                   subprocess.CompletedProcess([], 0, b"", b"")]
        with patch.object(mimo, "_macos"), patch.object(mimo.subprocess, "run", side_effect=replies), \
                patch.object(mimo, "read_saved_key", return_value="tp-other-dummy-key"):
            with self.assertRaises(mimo.CredentialError) as error:
                mimo.save_key_from_clipboard()
        self.assertNotIn(DUMMY, str(error.exception))
        self.assertNotIn("tp-other-dummy-key", str(error.exception))

    def test_keychain_errors_do_not_echo_command_output(self):
        reply = subprocess.CompletedProcess([], 1, DUMMY.encode(), DUMMY.encode())
        with patch.object(mimo, "_macos"), patch.object(mimo.subprocess, "run", return_value=reply):
            with self.assertRaises(mimo.CredentialError) as error:
                mimo.read_saved_key()
        self.assertNotIn(DUMMY, str(error.exception))

    def test_launcher_only_exposes_secret_in_child_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "previous-value"}), \
                patch.object(mimo, "read_saved_key", return_value=DUMMY), \
                patch.object(mimo.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(mimo.main(["--input", "captions.xlsx"]), 0)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "previous-value")
        command = run.call_args.args[0]
        self.assertNotIn(DUMMY, command)
        self.assertEqual(run.call_args.kwargs["env"]["OPENAI_API_KEY"], DUMMY)
        self.assertEqual(command[command.index("--base-url") + 1], mimo.BASE_URL)
        self.assertEqual(command[command.index("--timeout") + 1], "180")
        self.assertEqual(command[-2:], ["--input", "captions.xlsx"])

    def test_dry_run_does_not_read_keychain(self):
        with patch.object(mimo, "read_saved_key") as read, \
                patch.object(mimo.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(mimo.main(["--dry-run"]), 0)
        read.assert_not_called()

    def test_saved_key_cannot_be_redirected_to_another_endpoint(self):
        with contextlib.redirect_stderr(io.StringIO()), patch.object(mimo, "read_saved_key") as read:
            with self.assertRaises(SystemExit) as error:
                mimo.main(["--base-url", "https://other.invalid/v1"])
        self.assertEqual(error.exception.code, 2)
        read.assert_not_called()

    def test_save_only_does_not_launch_agent(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(mimo, "save_key_from_clipboard") as save, \
                patch.object(mimo.subprocess, "run") as run:
            self.assertEqual(mimo.main(["--save-key"]), 0)
        save.assert_called_once_with()
        run.assert_not_called()
        self.assertNotIn(DUMMY, output.getvalue())

    def test_check_only_never_displays_saved_key(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(mimo, "read_saved_key", return_value=DUMMY), \
                patch.object(mimo.subprocess, "run") as run:
            self.assertEqual(mimo.main(["--check-key"]), 0)
        run.assert_not_called()
        self.assertNotIn(DUMMY, output.getvalue())


if __name__ == "__main__":
    unittest.main()
