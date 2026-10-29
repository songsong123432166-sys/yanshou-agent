#!/usr/bin/env python3
"""macOS MiMo launcher using the current user's encrypted login Keychain.

Save copied key: .venv/bin/python run_mimo.py --save-key
Check saved key: .venv/bin/python run_mimo.py --check-key
Run:            .venv/bin/python run_mimo.py --input captions.xlsx
"""
from __future__ import annotations

import argparse
import hmac
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SERVICE = "CaptionQAAgent.MiMo"
ACCOUNT = "token-plan-cn.xiaomimimo.com"
BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL = "mimo-v2.6-pro"
SECURITY = "/usr/bin/security"
KEY_RE = re.compile(r"tp-[A-Za-z0-9_-]+")


class CredentialError(Exception):
    """Safe-to-display credential error; never contains a secret value."""


def _macos() -> None:
    if sys.platform != "darwin":
        raise CredentialError("此入口使用 macOS 钥匙串；Windows 请使用 run_mimo.ps1。")


def read_saved_key() -> str:
    _macos()
    result = subprocess.run(
        [SECURITY, "find-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w"],
        capture_output=True, timeout=60, check=False,
    )
    if result.returncode != 0:
        raise CredentialError("未找到可读取的 MiMo 密钥；请复制完整密钥并运行 --save-key，或解锁登录钥匙串。")
    key = result.stdout.decode("utf-8").strip()
    if not KEY_RE.fullmatch(key):
        raise CredentialError("钥匙串中的 MiMo 密钥格式无效；请重新复制完整密钥后运行 --save-key。")
    return key


def save_key_from_clipboard() -> None:
    _macos()
    result = subprocess.run(["/usr/bin/pbpaste"], capture_output=True, timeout=10, check=False)
    if result.returncode != 0:
        raise CredentialError("无法读取剪贴板；未保存密钥。")
    key = re.sub(r"^Bearer\s+", "", result.stdout.decode("utf-8").strip(), flags=re.IGNORECASE)
    if not KEY_RE.fullmatch(key):
        raise CredentialError("剪贴板不是完整的 MiMo Token Plan 密钥；未保存任何内容。")
    # security's interactive mode receives the secret over stdin, never argv.
    # Token validation excludes quotes, newlines and command separators.
    command = f'add-generic-password -U -s {SERVICE} -a {ACCOUNT} -l "CaptionQAAgent MiMo API key" -w "{key}"\n'
    result = subprocess.run(
        [SECURITY, "-i"], input=command.encode("utf-8"), capture_output=True, timeout=60, check=False,
    )
    if result.returncode != 0:
        raise CredentialError("写入钥匙串失败；请确认登录钥匙串已解锁。未显示密钥或命令输出。")
    if not hmac.compare_digest(key, read_saved_key()):
        raise CredentialError("钥匙串保存后的读取校验失败；未显示任何密钥。")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save-key", action="store_true", help="将剪贴板密钥保存/更新到 macOS 钥匙串，随后退出")
    group.add_argument("--check-key", action="store_true", help="确认已保存密钥可读取，不显示密钥、不调用 API")
    args, caption_args = parser.parse_known_args(argv)
    if (args.save_key or args.check_key) and caption_args:
        parser.error("--save-key/--check-key 单独使用；质检参数请在另一次运行时传入")
    if any(arg == "--base-url" or arg.startswith("--base-url=") for arg in caption_args):
        parser.error("此入口固定使用 MiMo Token Plan 中国地址，不支持覆盖 --base-url")
    try:
        if args.save_key:
            save_key_from_clipboard()
            print("MiMo 密钥已保存到 macOS 钥匙串，并已读取校验成功。未写入项目文件。")
            return 0
        if args.check_key:
            read_saved_key()
            print("已保存的 MiMo 密钥可正常读取。未显示密钥，未调用 API。")
            return 0
        environment = os.environ.copy()
        if "--dry-run" not in caption_args:
            environment["OPENAI_API_KEY"] = read_saved_key()
        command = [sys.executable, str(Path(__file__).with_name("caption_qa_agent.py")),
                   "--base-url", BASE_URL, "--model", MODEL, "--timeout", "180", *caption_args]
        # The saved key is supplied only in this child process's environment.
        return subprocess.run(command, env=environment, check=False).returncode
    except CredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except (OSError, UnicodeError, subprocess.SubprocessError):
        # Never echo subprocess inputs/outputs or exception bodies here.
        print("MiMo 启动或钥匙串操作失败；未显示任何密钥。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
