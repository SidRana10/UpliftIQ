import numpy as np
import pandas as pd
import pytest

from upliftiq.data.registry import HILLSTROM, RAW_DIR
from upliftiq.evaluation.hillstrom_analysis import (
    balance_table,
    group_summary,
    treatment_effects,
)


def _synthetic(n_per_arm: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for arm, p in [("No E-Mail", 0.05), ("Mens E-Mail", 0.12), ("Womens E-Mail", 0.09)]:
        visit = (rng.random(n_per_arm) < p).astype(int)
        conv = visit * (rng.random(n_per_arm) < 0.2).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "recency": rng.integers(1, 13, n_per_arm),
                    "history_segment": rng.choice(["a", "b", "c"], n_per_arm),
                    "history": rng.exponential(100, n_per_arm),
                    "mens": rng.integers(0, 2, n_per_arm),
                    "womens": rng.integers(0, 2, n_per_arm),
                    "zip_code": rng.choice(["Urban", "Rural"], n_per_arm),
                    "newbie": rng.integers(0, 2, n_per_arm),
                    "channel": rng.choice(["Web", "Phone"], n_per_arm),
                    "segment": arm,
                    "visit": visit,
                    "conversion": conv,
                    "spend": conv * rng.exponential(80, n_per_arm),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_group_summary_counts():
    s = group_summary(_synthetic())
    assert s["n"].tolist() == [3000, 3000, 3000]
    assert s["share"].sum() == pytest.approx(1.0)


def test_treatment_effects_shape_and_direction():
    e = treatment_effects(_synthetic())
    assert len(e) == 6
    assert (e.loc[e["outcome"] == "visit", "abs_effect"] > 0).all()
    assert (e["p_value_holm"] >= e["p_value"]).all()
    assert e.loc[e["outcome"] == "visit", "significant_holm_0.05"].all()


def test_treatment_effects_reproducible():
    df = _synthetic()
    pd.testing.assert_frame_equal(treatment_effects(df), treatment_effects(df))


def test_balance_table_on_randomized_data():
    b = balance_table(_synthetic())
    assert len(b) == 8
    assert b["balanced_smd_lt_0.1"].all()


@pytest.mark.network
def test_real_data_known_results():
    path = RAW_DIR / HILLSTROM.filename
    if not path.exists():
        pytest.skip("Hillstrom not downloaded")
    df = pd.read_csv(path)
    s = group_summary(df).set_index("segment")
    assert s.loc["Mens E-Mail", "visit_rate"] == pytest.approx(0.1828, abs=1e-3)
    assert s.loc["No E-Mail", "conversion_rate"] == pytest.approx(0.0057, abs=1e-3)
    assert balance_table(df)["balanced_smd_lt_0.1"].all()
