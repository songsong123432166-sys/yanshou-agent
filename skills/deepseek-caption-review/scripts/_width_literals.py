"""Portable offline rule support; no application or API imports."""
import re


SCREEN = re.compile(r'【[^【】]*(?:】|$)')

TECH = re.compile(r'(?:https?://|www\.)[^\s<＜【｛{，。；！？、（）]+|[\w.+-]+@[\w.-]+\.[A-Za-z]+|'
                  r'(?<!\w)\d+(?:(?:[.:/-])\d+)+(?!\w)|'
                  r'(?<!\w)(?:[A-Za-z]:[\\/]|/|\./|\.\./)[\w./\\:+?=&%-]+|'
                  r'\b[A-Za-z_][\w.-]*\.[A-Za-z]{1,8}\b|`[^`]*`')
