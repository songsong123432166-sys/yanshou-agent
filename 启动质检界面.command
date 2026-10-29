#!/bin/zsh
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print '请先双击安装环境.command。需要AI质检时，再运行配置MiMo密钥.command。'
  read 'reply?按回车关闭…'
  exit 1
fi
exec .venv/bin/python agent.py serve --open
