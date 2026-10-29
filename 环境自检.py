#!/usr/bin/env python3
"""Check local dependencies and portable exports without keys or network."""
import importlib
from pathlib import Path
import sys
import tempfile


def main():
    if sys.version_info<(3,10):
        print('需要 Python 3.10 或更新版本。');return 1
    try:
        for name in ('openai','openpyxl','captionqa.caption_qa_web','captionqa.caption_field_jobs','captionqa.caption_qa_portable_exports'):
            importlib.import_module(name)
        from captionqa.caption_qa_web import DEFAULT_OUTPUT_DIR
        from captionqa.caption_qa_portable_exports import repair_xlsx
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'self-check.xlsx'
            repair_xlsx(path,[['0001','合成自检记录']],True,True,[])
            book=load_workbook(path)
            try:assert book.active['A2'].value=='0001'
            finally:book.close()
        print('Python、模型SDK、Excel读写、网页模块及字段整理模块均可用。')
        print('输出目录：'+str(DEFAULT_OUTPUT_DIR))
        print('本次未读取密钥，未调用API，未处理业务文件。')
        return 0
    except (ImportError,OSError,ValueError,AssertionError) as exc:
        print('环境自检未通过：'+type(exc).__name__+'；请重跑安装环境.command或联系维护人。');return 1


if __name__=='__main__':raise SystemExit(main())
