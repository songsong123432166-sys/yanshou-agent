#!/usr/bin/env python3
"""Unified entry: launch (default), serve, check or mimo."""
import sys


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    mode=args.pop(0) if args and args[0] in {'launch','serve','check','mimo'} else 'launch'
    if mode=='serve':
        from captionqa.caption_qa_web import main as run
    elif mode=='check':
        from captionqa.caption_qa_agent import main as run, DEMO_CAPTION
        return run(DEMO_CAPTION, argv=args)
    elif mode=='mimo':
        from captionqa.run_mimo import main as run
    else:
        from captionqa.caption_qa_launcher import main as run
    return run(args)


if __name__=='__main__':raise SystemExit(main())
