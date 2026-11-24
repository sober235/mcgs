#!/usr/bin/env python3
"""Start a shell command fully detached from the Claude Code process (new session, stdin from /dev/null), so that
a reload of the editor window does not end it. Usage: detach.py LOGFILE -- SHELL_COMMAND  (output appended to LOGFILE)"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[3]
if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    os._exit(0)
fd = os.open("/dev/null", os.O_RDONLY)
os.dup2(fd, 0)
out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(out, 1)
os.dup2(out, 2)
os.execvp("/bin/zsh", ["/bin/zsh", "-c", cmd])
