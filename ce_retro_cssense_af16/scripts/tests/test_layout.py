from pathlib import Path


def test_topic_symlinks_point_into_worktree():
    topic = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
    wt = Path("/data0/congcong/code/GS/mcgs_ce_retro_wt/ce_retro_cssense_af16")
    for name in ("scripts", "configs"):
        assert (topic / name).is_symlink()
        assert (topic / name).resolve() == (wt / name).resolve()
