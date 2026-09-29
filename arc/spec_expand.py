"""spec_expand: write one spec per subject from a template (spec 2026-09-28 §2.2).

Every string value of the template that contains "{subject}" gets the subject id; nothing else changes,
so all per-subject specs of one arm carry the same hyper-parameters. differs_only_by_subject() is what the
dataset branches' tests use to prove it. A "{subject}" inside a mapping key is refused (it would not be
substituted).

CLI:  python arc/spec_expand.py TEMPLATE OUT_DIR STEM SUBJECT [SUBJECT ...]   -> OUT_DIR/STEM_<SUBJECT>.yaml
      Every expanded spec must pass spec_io.validate_spec, and no target file may exist; otherwise nothing
      is written.
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


def _keys_with_placeholder(node, path: str = "") -> list:
    if isinstance(node, dict):
        found = []
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(k, str) and PLACEHOLDER in k:
                found.append(p)
            found += _keys_with_placeholder(v, p)
        return found
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _keys_with_placeholder(v, f"{path}[{i}]")]
    return []


def expand(template: dict, subject: str) -> dict:
    bad = _keys_with_placeholder(template)
    if bad:
        raise ValueError(f"{PLACEHOLDER} inside a mapping key is not substituted; move it into a value: {bad}")
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
    from spec_io import SpecError, validate_spec     # arc/ is on sys.path both as a script and in the tests
    template = yaml.safe_load(Path(a.template).read_text())
    out = Path(a.out_dir)
    specs = {s: expand(template, s) for s in a.subjects}
    for s, spec in specs.items():
        try:
            validate_spec(spec)
        except SpecError as e:
            raise SystemExit(f"the spec expanded for {s} is invalid (nothing written): {e}")
    existing = [str(out / f"{a.stem}_{s}.yaml") for s in specs if (out / f"{a.stem}_{s}.yaml").exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing spec file(s) (nothing written): {existing}")
    out.mkdir(parents=True, exist_ok=True)
    for s, spec in specs.items():
        (out / f"{a.stem}_{s}.yaml").write_text(yaml.safe_dump(spec, sort_keys=True))
        print(f"wrote {out / f'{a.stem}_{s}.yaml'}")


if __name__ == "__main__":
    main()
