#!/usr/bin/env python3
"""Remove the server clock from job folder names in files that are committed (time rule: no server clock value in a
log). A job folder is `j-<date>T<time>-<suffix>`; only `j-<suffix>` is kept. Run before every commit of an
experiments.jsonl:  scrub_clock.py FILE [FILE ...]   (prints the number of changed lines per file)."""
import pathlib
import re
import sys

CLOCK = re.compile(r"j-[0-9]{8}T[0-9]{6}-")


def scrub(text):
    return CLOCK.sub("j-", text)


def scrub_file(path):
    """Rewrite the file in place; returns the number of lines that changed."""
    path = pathlib.Path(path)
    lines = path.read_text().splitlines(keepends=True)
    out = [scrub(line) for line in lines]
    n = sum(a != b for a, b in zip(lines, out))
    if n:
        path.write_text("".join(out))
    return n


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(scrub_file(name), name)
