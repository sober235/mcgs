import pytest
from crisp.data.subjects import split_subjects, token_map

KEYS = [f"subj_{i:02d}" for i in range(16)]

def test_split_sizes_and_forced_member():
    s = split_subjects(KEYS, forced_dev="subj_07", n_dev=5)
    assert len(s["development"]) == 5
    assert len(s["test"]) == 11
    assert "subj_07" in s["development"]
    assert set(s["development"]) & set(s["test"]) == set()
    assert set(s["development"]) | set(s["test"]) == set(KEYS)

def test_split_is_order_invariant_and_deterministic():
    a = split_subjects(KEYS, forced_dev="subj_07", n_dev=5)
    b = split_subjects(list(reversed(KEYS)), forced_dev="subj_07", n_dev=5)
    assert a == b
    assert a == split_subjects(KEYS, forced_dev="subj_07", n_dev=5)

def test_forced_member_must_exist():
    with pytest.raises(ValueError):
        split_subjects(KEYS, forced_dev="not_a_subject", n_dev=5)

def test_tokens_are_stable_and_unique():
    t = token_map(KEYS)
    assert len(set(t.values())) == 16
    assert t == token_map(list(reversed(KEYS)))
    assert all(v.startswith("S") and len(v) == 4 for v in t.values())
