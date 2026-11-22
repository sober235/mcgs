"""Study profiles of the local queue (tools/queue.py): where a study's launcher, server root, local result file and
training code are, and how a row is written. No third-party imports and no gpu.py import, so the tests can load it.

    v1    the first version (motion adapter)          rows af:cond:arm
    mild  the second version (three small adjustments) rows af:order:cond:arm
"""
V1 = "autoresearch/motion_adapter_sim_20261113"
MILD = "autoresearch/motion_mild_adjust_20261121"
BASE = "/data0/congcong/code/GS/3DGS/outputs/codex_artifacts"

STUDIES = {
    "v1": dict(rel=V1, launcher=f"{V1}/run.py", root=f"{BASE}/mcgs_motion_adapter_20261113", prefix="ma", fields=3,
               code=("adapter.py", "common.py", "geom.py", "patch.py", "run.py", "schedule.py", "warp.py")),
    # code paths of the second version are relative to the repository root: its own files and the library it uses
    "mild": dict(rel=MILD, launcher=f"{MILD}/mrun.py", root=f"{BASE}/mcgs_motion_mild_adjust_20261121", prefix="mm",
                 fields=4,
                 code=tuple(f"{MILD}/{f}" for f in ("mcommon.py", "mpatch.py", "mrun.py", "weights.py", "align.py",
                                                    "reg.py", "shiftstat.py"))
                 + tuple(f"{V1}/{f}" for f in ("adapter.py", "common.py", "geom.py", "patch.py", "run.py",
                                               "schedule.py", "warp.py", "probe_reginit.py"))),
}


def parse_row(study, text):
    row = tuple(text.split(":"))
    assert len(row) == STUDIES[study]["fields"], f"{study} rows have {STUDIES[study]['fields']} fields: {text!r}"
    return row


def exp_name(study, row, seed, tag=""):
    return f"{STUDIES[study]['prefix']}_" + "_".join(row) + ("" if seed == 42 else f"_s{seed}") + (f"_{tag}" if tag else "")


def launch_argv(study, row, card, seed, settings, tag):
    if study == "v1":
        af, cond, arm = row
        head = ["--af", af, "--cond", cond, "--arm", arm]
    else:
        af, order, cond, arm = row
        head = ["--af", af, "--order", order, "--cond", cond, "--arm", arm]
    return ["@python", STUDIES[study]["launcher"], *head, "--gpu", str(card), "--seed", str(seed)] \
        + (["--settings", settings] if settings else []) + (["--tag", tag] if tag else [])
