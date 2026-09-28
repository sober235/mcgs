"""Write specs/brain_hp_n36_equiv.yaml: the hp_n36 run (jyz AF16 gs_joint_p3 anchor, head PSNR 35.706 / 34.928)
as a run_spec spec that only scores the saved checkpoint. Used to show spec_scoring == cycle.py (plan Task 5).
Source = the resolved config JointTrainer wrote for that run. Output section and seed handling follow cycle.py:
output_dir / exp_name are set by run_spec.py, and cycle.py seeds with set_seed(42, deterministic=False)."""
from pathlib import Path

import yaml

HP = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36")
ROOT = Path(__file__).resolve().parents[1]

trainer = yaml.safe_load((HP / "config.yaml").read_text())
trainer.setdefault("output", {})
trainer["output"].pop("output_dir", None)
trainer["output"].pop("exp_name", None)
trainer["training"].setdefault("seed", 42)
trainer["training"].setdefault("deterministic", False)
spec = {"spec": {"dataset": "brain_jyz_af16_hp_n36", "exp_prefix": "brainpin",
                 "results_root": str(ROOT / "results_spec"), "save_volumes": False,
                 "from_ckpt": str(HP / "checkpoints/latest.pth")},
        "scoring": {"anatomy_rel": 0.05, "grain_ratio": True},
        "trainer": trainer}
out = ROOT / "specs/brain_hp_n36_equiv.yaml"
out.parent.mkdir(exist_ok=True)
out.write_text(yaml.safe_dump(spec, sort_keys=True))
print(f"wrote {out}")
