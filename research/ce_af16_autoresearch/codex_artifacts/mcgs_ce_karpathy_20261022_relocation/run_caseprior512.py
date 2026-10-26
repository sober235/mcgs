#!/usr/bin/env python3
"""Stdlib entry point for the fixed512 case-prior protocol."""
from caseprior_budget_variant import load_protocol


def parser():
    return load_protocol().parser()


def main(argv=None):
    return load_protocol().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
