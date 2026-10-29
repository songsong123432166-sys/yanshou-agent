#!/bin/zsh
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print '请先双击安装环境.command。'
  read 'reply?按回车关闭…'
  exit 1
fi
print '请复制你自己的完整MiMo Token Plan密钥到剪贴板，复制后不要再复制其他内容。'
print '按回车后只保存到当前Mac用户的系统钥匙串，不显示密钥、不调用API。'
read 'reply?准备好后按回车；取消请按 Ctrl+C…'
.venv/bin/python run_mimo.py --save-key
result=$?
read 'reply?按回车关闭…'
exit "$result"
