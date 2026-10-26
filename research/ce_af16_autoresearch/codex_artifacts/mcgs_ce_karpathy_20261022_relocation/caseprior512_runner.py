#!/usr/bin/env python3
"""Child entry point for the fixed512 original-trainer continuation."""
import argparse
from pathlib import Path

from caseprior_budget_variant import load_child, load_protocol


def execute(path):
    return load_child().execute(path)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        execute(args.launch)
    except BaseException as error:
        try:
            import os
            child = load_child()
            directory = Path(load_protocol().base.read(args.launch)['run_dir'])
            child.g2.write_json(directory/'numeric_failure.json',
                dict(status='failed', error=repr(error), pid=os.getpid()))
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
