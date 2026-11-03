import numpy as np
import pandas as pd
import pytest

from upliftiq.data.registry import HILLSTROM, RAW_DIR
from upliftiq.modeling.split import TEST, Split, load_split, make_split, save_split


def _frame(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
    visit = (rng.random(n) < np.where(seg == "No E-Mail", 0.1, 0.17)).astype(int)
    return pd.DataFrame({"segment": seg, "visit": visit})


def test_split_is_reproducible_and_seed_sensitive():
    df = _frame()
    a, b = make_split(df), make_split(df)
    assert (a.assignment == b.assignment).all()
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != make_split(df, seed=7).fingerprint()


def test_split_partitions_all_rows_exactly_once():
    df = _frame()
    s = make_split(df)
    assert len(s.test_idx) == pytest.approx(0.2 * len(df), abs=2)
    assert len(s.dev_idx) + len(s.test_idx) == len(df)
    assert set(np.unique(s.assignment)) == {TEST, 0, 1, 2, 3, 4}
    sizes = [len(s.fold_indices(k)[1]) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(s.dev_idx)


def test_folds_are_disjoint_and_never_touch_test():
    s = make_split(_frame())
    test = set(s.test_idx)
    for k in range(5):
        train, val = s.fold_indices(k)
        assert not set(train) & set(val)
        assert not (set(train) | set(val)) & test
        assert set(train) | set(val) == set(s.dev_idx)
    with pytest.raises(ValueError):
        s.fold_indices(5)


def test_stratification_preserves_arm_and_visit_rates():
    df = _frame(20000)
    s = make_split(df)
    for idx in (s.test_idx, *(s.fold_indices(k)[1] for k in range(5))):
        part = df.iloc[idx]
        assert part["visit"].mean() == pytest.approx(df["visit"].mean(), abs=0.003)
        for arm, share in df["segment"].value_counts(normalize=True).items():
            assert (part["segment"] == arm).mean() == pytest.approx(share, abs=0.005)


def test_save_load_roundtrip_and_validation(tmp_path):
    s = make_split(_frame())
    path = tmp_path / "split.json"
    save_split(s, path)
    loaded = load_split(path, n_rows=s.n_rows)
    assert (loaded.assignment == s.assignment).all()
    with pytest.raises(ValueError):
        load_split(path, n_rows=s.n_rows + 1)
    path.write_text(path.read_text().replace('"assignment": [', '"assignment": [0, ', 1))
    with pytest.raises(ValueError):
        load_split(path)
    assert isinstance(loaded, Split)


@pytest.mark.network
def test_real_data_split_sizes_and_balance():
    path = RAW_DIR / HILLSTROM.filename
    if not path.exists():
        pytest.skip("Hillstrom not downloaded")
    df = pd.read_csv(path)
    s = make_split(df)
    assert len(s.test_idx) == 12_800
    assert len(s.dev_idx) == 51_200
    assert all(len(s.fold_indices(k)[1]) == 10_240 for k in range(5))
    test = df.iloc[s.test_idx]
    assert test["visit"].mean() == pytest.approx(df["visit"].mean(), abs=0.001)
    for arm, share in df["segment"].value_counts(normalize=True).items():
        assert (test["segment"] == arm).mean() == pytest.approx(share, abs=0.001)
