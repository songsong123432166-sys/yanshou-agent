#!/bin/zsh
cd -- "${0:A:h}"
fail() { print -- "$1"; read 'reply?按回车关闭…'; exit 1; }
python_executable=''
for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
    python_executable="$(command -v "$candidate")"
    break
  fi
done
[[ -n "$python_executable" ]] || fail '请先安装 Python 3.10 或更新版本，再重新运行本文件。不需要安装 Node.js。'
"$python_executable" -m venv .venv || fail '创建独立 Python 环境失败，请确认文件夹可写。'
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-share.txt || fail '依赖安装失败。请检查网络后重试；不要使用 sudo。'
.venv/bin/python 环境自检.py || fail '环境自检未通过，请查看上面的提示。'
print '环境安装完成。需要AI质检时，请先配置你自己的MiMo密钥；然后双击启动质检界面.command。'
read 'reply?按回车关闭…'
