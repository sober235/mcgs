"""Spec 5.6: BART pics per echo on a named spoke set (r0, low, high, ref, af8, af12, af16)."""
import argparse
import json

from common import prepare_inputs, recon_dir, ref_lmax, run_many

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--set", required=True)
ap.add_argument("--echoes", default="0,1,2,3,4,5,6,7")
ap.add_argument("--reg", default="l2", choices=["l2", "l1"])
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--tag", default=None)
ap.add_argument("--w-from-ref", action="store_true", help="-w = REF scaling of the same echo")
ap.add_argument("--debug", action="store_true", help="-d5; a REF run records BART's scaling")
ap.add_argument("--lam-rel", action="store_true", help="--lam is relative to the REF operator's largest eigenvalue (spec 5.6, 2026-09-11)")
ap.add_argument("--jobs", type=int, default=1)
a = ap.parse_args()
echoes = [int(e) for e in a.echoes.split(",")]
tag = a.tag or a.set
prepare_inputs(a.subject, a.set, echoes)
scal = json.loads((recon_dir(a.subject) / "ref_scaling.json").read_text()) if a.w_from_ref else {}
lam = a.lam * ref_lmax(a.subject) if a.lam_rel else a.lam
jobs = [dict(subject=a.subject, name=a.set, echo=e, reg=a.reg, lam=lam, lam_rel=a.lam if a.lam_rel else None, iters=a.iters, tag=tag,
             w=scal[str(e)] if a.w_from_ref else None, debug=a.debug) for e in echoes]
infos = run_many(jobs, a.jobs)
if a.debug and a.set == "ref" and tag == "ref":
    p = recon_dir(a.subject) / "ref_scaling.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update({str(i["echo"]): i["scaling"] for i in infos})
    p.write_text(json.dumps(old, indent=1))
for i in infos:
    print(f"{i['tag']} e{i['echo']}: {i['n_spokes']} spokes, {i['seconds']:.0f} s, scaling={i['scaling']}", flush=True)
