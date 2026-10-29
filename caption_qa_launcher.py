#!/usr/bin/env python3
"""Open the local Agent, reusing a live service or starting it in the background."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler,Request,build_opener

ROOT=Path(__file__).resolve().parent


class LaunchError(Exception):pass


def probe(port):
    """Do not read credentials/config or mistake another service for this Agent."""
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.6):pass
    except ConnectionRefusedError:return 'absent'
    except OSError:return 'occupied'
    opener=build_opener(ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/',timeout=1.5) as response:
            page=response.read(256*1024).decode('utf-8')
        token=re.search(r'<meta name="app-token" content="([A-Za-z0-9_-]+)"',page)
        if not token or 'Caption QA' not in page or '/app.js' not in page:return 'occupied'
        request=Request(f'http://127.0.0.1:{port}/api/jobs',headers={'X-App-Token':token[1]})
        with opener.open(request,timeout=1.5) as response:
            import json
            data=json.loads(response.read())
        return 'ready' if isinstance(data,dict) and isinstance(data.get('jobs'),list) else 'occupied'
    except (OSError,URLError,UnicodeError,ValueError):return 'occupied'


def open_page(url):
    if sys.platform=='darwin':
        try:subprocess.run(['/usr/bin/open',url],check=True,capture_output=True,timeout=10)
        except (OSError,subprocess.SubprocessError):raise LaunchError('服务已启动，但浏览器没有打开。请直接访问 '+url) from None
    else:
        import webbrowser
        if not webbrowser.open(url):raise LaunchError('请在浏览器打开 '+url)


def launch(port=8765,*,open_browser=True,timeout=20,root=ROOT):
    root=Path(root);folder=root/'.local/launcher';folder.mkdir(parents=True,exist_ok=True)
    url=f'http://127.0.0.1:{port}';log_path=folder/'server.log'
    with (folder/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=probe(port)
        if state=='ready':
            if open_browser:open_page(url)
            return '已打开正在运行的 Caption 质检界面：'+url
        if state!='absent':raise LaunchError(f'{port}端口被其他程序或未就绪的服务占用，未停止任何进程。请稍后重试或查看 {log_path}')
        if not (root/'caption_qa_web.py').is_file():raise LaunchError('找不到Agent主程序，请恢复完整项目文件夹。')
        if any(importlib.util.find_spec(name) is None for name in ('openai','openpyxl')):
            raise LaunchError('运行环境缺少依赖，请先按README安装requirements.txt。')
        try:
            with log_path.open('ab') as log:
                child=subprocess.Popen([sys.executable,str(root/'caption_qa_web.py'),'--port',str(port)],
                    cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,close_fds=True)
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if probe(port)=='ready':
                    (folder/'server.pid').write_text(str(child.pid)+'\n',encoding='ascii')
                    if open_browser:open_page(url)
                    return 'Caption Agent 已在后台启动：'+url
                if child.poll() is not None:raise LaunchError('服务启动未完成，请查看日志：'+str(log_path))
                time.sleep(.25)
        except OSError:raise LaunchError('启动后台服务失败，请查看日志：'+str(log_path)) from None
        raise LaunchError('服务还未就绪，请稍后再次双击入口；不会终止后台任务。日志：'+str(log_path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-open',action='store_true',help='只检查/启动服务，不打开浏览器')
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535:parser.error('端口须在1至65535之间。')
    try:print(launch(args.port,open_browser=not args.no_open));return 0
    except (LaunchError,OSError) as exc:print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
