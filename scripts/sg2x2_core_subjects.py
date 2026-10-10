#!/usr/bin/env python3
"""The sg2x2 core-test subjects and their file paths.

Source: the data inventory of 2026-10-10 (docs/superpowers/records/2026-10-10-sg2x2-core-test-data-inventory.md).
Fourteen regular T1 + FLAIR pairs of the UIH 3D brain set: jiangyizhou_20230705 is the development subject (the
pilot's subject, here in its h5 export), the other thirteen are test subjects. The two `790` files are excluded: they
are second scans of jiangdian_20230705 and tangbing_20230705 on another protocol (inventory sections 2 and 5).
"""
from __future__ import annotations

SRC = {"T1": "/data1/data1_congcong/data/3D/T1_csm_ksp", "FLAIR": "/data1/data1_congcong/data/3D/FLAIR_csm_ksp"}
OUT_ROOT = "/data2/congcong/data/sg2x2_core_h5"
CONTRASTS = ("T1", "FLAIR")

SUBJECTS = {
    "jiangyizhou_20230705": ("Train_3dBrain_jiangyizhou_20230705-152818-1450_152931.h5", "dev"),
    "huangyuwei_20230705": ("Train_3dBrain_huangyuwei_20230705-160927-1451_161015.h5", "test"),
    "jiangdian_20230705": ("Train_3dBrain_jiangdian_20230705-144538-1449_144607.h5", "test"),
    "sunxinyi_20230705": ("Train_3dBrain_sunxinyi_20230705-164717-1452_164752.h5", "test"),
    "tangbing_20230705": ("Train_3dBrain_tangbing_20230705-132343-1448_132439.h5", "test"),
    "liaozhenghan_20231030": ("Train_3dBrain_liaozhenghan_20231030-141603-1576_141649.h5", "test"),
    "zhoushuo_20231030": ("Train_3dBrain_zhoushuo_20231030-161805-1579_161837.h5", "test"),
    "shiyixuan_20231101": ("Train_3dBrain_shiyixuan_20231101-163220-1587_163304.h5", "test"),
    "zhangyangming_20231101": ("Train_3dBrain_zhangyangming_20231101-144341-1584_144414.h5", "test"),
    "yangkun_20231102": ("Train_3dBrain_yangkun_20231102-131858-1588_131925.h5", "test"),
    "zhangxiqian_20231102": ("Train_3dBrain_zhangxiqian_20231102-142131-1589_143120.h5", "test"),
    "chenshuaibao_20231106": ("Train_3dBrain_chenshuaibao_20231106-114149-1598_114223.h5", "test"),
    "lishengbo_20231106": ("Train_3dBrain_lishengbo_20231106-154803-1601_154834.h5", "test"),
    "litianyu_20231107": ("Train_3dBrain_litianyu_20231107-131010-1602_131057.h5", "test"),
}
EXCLUDED = {
    "790_jiangdian_20230706": "second scan of jiangdian_20230705 on another protocol (50 % slab oversampling)",
    "790_tangbin_20230706": "second scan of tangbing_20230705 on another protocol (50 % slab oversampling)",
}


def core_test_subjects() -> list[str]:
    return [s for s, (_, role) in SUBJECTS.items() if role == "test"]


def subject_paths(subject: str, out_root: str = OUT_ROOT) -> dict:
    """the subject's two source h5 files and the export's output files"""
    fname, role = SUBJECTS[subject]
    out = f"{out_root}/{subject}"
    return {"role": role, "src": {c: f"{SRC[c]}/{fname}" for c in CONTRASTS}, "out": out,
            "ksp": {c: f"{out}/ksp_{c}.h5" for c in CONTRASTS},
            "csm": {c: f"{out}/csm_espirit_{c}.h5" for c in CONTRASTS},
            "bart": f"{out}/bart", "report": f"{out}/export.json"}
