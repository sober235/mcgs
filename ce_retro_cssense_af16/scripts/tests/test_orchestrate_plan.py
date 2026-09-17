import json


def test_job_graph_covers_all_arms_and_extends_edge_lambda(tiny_topic):
    import retro_orchestrate as O
    top = tiny_topic["root"]
    jobs = O.build_jobs(top)
    names = {j["name"] for j in jobs}
    for n in ("cls_pre", "cls_post", "cs_pre_lam0.001", "joint_lam0.001", "spirit_post_ks3_lam0.002", "zs_pre", "dip_post", "inr_pre",
              "mcinr", "gs_single_pre", "gs_single_post", "gs_joint", "summarize"):
        assert n in names, n
    cmd = next(j for j in jobs if j["name"] == "cs_pre_lam0.001")["cmd"](3)
    assert "--admm_outer" in cmd and cmd[cmd.index("--admm_outer") + 1] == "200" and "--gpu" in cmd and cmd[cmd.index("--gpu") + 1] == "3"
    assert next(j for j in jobs if j["name"] == "summarize")["deps"]
    # edge extension: best lambda sits at the top of the grid -> one more job at 1.4x
    (top / "logs").mkdir(exist_ok=True)
    for lam in O.GRID["cs"]:
        json.dump({"rows": [dict(method=f"cssense_o200_lam{lam:g}", contrast="pre", head_psnr=30 + lam * 1e3, lam=lam)]},
                  open(top / f"logs/classical_pre_o200_lam{lam:g}.json", "w"))
    ext = O.plan_extensions(top, {j["name"]: dict(j, state="done") for j in jobs})
    assert [j["name"] for j in ext] == ["cs_pre_lam0.0028"]
