"""spec_expand: write one spec per subject from a template (spec 2026-09-28 §2.2).

Every string value of the template that contains "{subject}" gets the subject id; nothing else changes,
so all per-subject specs of one arm carry the same hyper-parameters. differs_only_by_subject() is what the
dataset branches' tests use to prove it.

CLI:  python arc/spec_expand.py TEMPLATE OUT_DIR STEM SUBJECT [SUBJECT ...]   -> OUT_DIR/STEM_<SUBJECT>.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PLACEHOLDER = "{subject}"


def _count(node) -> int:
    if isinstance(node, dict):
        return sum(_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count(v) for v in node)
    return int(isinstance(node, str) and PLACEHOLDER in node)


def _sub(node, subject: str):
    if isinstance(node, dict):
        return {k: _sub(v, subject) for k, v in node.items()}
    if isinstance(node, list):
        return [_sub(v, subject) for v in node]
    if isinstance(node, str):
        return node.replace(PLACEHOLDER, subject)
    return node


def expand(template: dict, subject: str) -> dict:
    if _count(template) == 0:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    return _sub(template, subject)


def differs_only_by_subject(template, expanded, subject: str) -> bool:
    if isinstance(template, dict):
        return (isinstance(expanded, dict) and set(template) == set(expanded)
                and all(differs_only_by_subject(template[k], expanded[k], subject) for k in template))
    if isinstance(template, list):
        return (isinstance(expanded, list) and len(template) == len(expanded)
                and all(differs_only_by_subject(t, e, subject) for t, e in zip(template, expanded)))
    if isinstance(template, str):
        return expanded == template.replace(PLACEHOLDER, subject)
    return type(template) is type(expanded) and template == expanded


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="expand a spec template per subject")
    ap.add_argument("template")
    ap.add_argument("out_dir")
    ap.add_argument("stem")
    ap.add_argument("subjects", nargs="+")
    a = ap.parse_args(argv)
    template = yaml.safe_load(Path(a.template).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        (out / f"{a.stem}_{s}.yaml").write_text(yaml.safe_dump(expand(template, s), sort_keys=True))
        print(f"wrote {out / f'{a.stem}_{s}.yaml'}")


if __name__ == "__main__":
    main()
