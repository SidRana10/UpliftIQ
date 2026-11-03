import numpy as np
import pandas as pd
import pytest

from upliftiq.modeling.features import (
    ARMS,
    FEATURES,
    FORBIDDEN,
    assert_no_leakage,
    encode_arm,
    make_preprocessor,
    prepare_dev_test,
    prepare_fold,
    select_features,
    tree_categorical_indices,
)
from upliftiq.modeling.split import make_split


def _frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
            "history": rng.exponential(200, n) + 30,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
            "segment": rng.choice(ARMS, n),
            "visit": rng.integers(0, 2, n),
            "conversion": rng.integers(0, 2, n),
            "spend": rng.exponential(50, n),
        }
    )


def test_feature_whitelist_is_the_seven_approved_features():
    assert sorted(FEATURES) == sorted(
        ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    )
    assert FORBIDDEN.isdisjoint(FEATURES)


def test_select_features_drops_outcomes_treatment_and_redundant_bin():
    X = select_features(_frame())
    assert list(X.columns) == FEATURES


@pytest.mark.parametrize("bad", ["visit", "conversion", "spend", "segment", "history_segment"])
def test_leakage_guard_rejects_forbidden_columns(bad):
    with pytest.raises(ValueError):
        assert_no_leakage([*FEATURES, bad])


def test_leakage_guard_rejects_missing_or_extra_columns():
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURES[:-1])
    with pytest.raises(ValueError):
        assert_no_leakage([*FEATURES, "something_else"])
    with pytest.raises(KeyError):
        select_features(_frame().drop(columns=["channel"]))


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_feature_matrix_is_independent_of_outcomes_and_treatment(kind):
    df = _frame()
    split = make_split(df)
    a = prepare_fold(df, split, 0, kind)
    rng = np.random.default_rng(1)
    shuffled = df.copy()
    for col in ["visit", "conversion", "spend", "segment"]:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    b = prepare_fold(shuffled, make_split(df), 0, kind)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.X_eval, b.X_eval)


def test_encode_arm_codes_and_unknown_arm():
    codes = encode_arm(pd.Series(["Womens E-Mail", "No E-Mail", "Mens E-Mail"]))
    assert codes.tolist() == [2, 0, 1]
    with pytest.raises(ValueError):
        encode_arm(pd.Series(["No E-Mail", "Kids E-Mail"]))


def test_prepared_fold_carries_arm_and_outcomes_outside_X():
    df = _frame()
    split = make_split(df)
    d = prepare_fold(df, split, 2)
    train_idx, val_idx = split.fold_indices(2)
    assert d.X_train.shape[0] == len(d.arm_train) == len(d.y_train) == len(train_idx)
    assert d.X_eval.shape[0] == len(d.arm_eval) == len(d.y_eval) == len(val_idx)
    assert list(d.y_train.columns) == ["visit", "conversion", "spend"]
    np.testing.assert_array_equal(d.arm_eval, encode_arm(df.iloc[val_idx]["segment"]))
    assert d.X_train.shape[1] == d.X_eval.shape[1]


def test_linear_preprocessor_fit_on_training_rows_only():
    df = _frame()
    split = make_split(df)
    d = prepare_fold(df, split, 0, "linear")
    train_idx, _ = split.fold_indices(0)
    train_hist = np.log1p(df.iloc[train_idx]["history"])
    scaler = d.preprocessor.named_transformers_["history"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(train_hist.mean())
    assert scaler.mean_[0] != pytest.approx(np.log1p(df["history"]).mean(), abs=1e-12)
    assert d.X_train[:, 0].mean() == pytest.approx(0, abs=1e-9)
    assert not np.isnan(d.X_train).any() and not np.isnan(d.X_eval).any()


def test_one_hot_columns_and_unseen_category_handling():
    df = _frame()
    split = make_split(df)
    train_idx, _ = split.fold_indices(0)
    pre = make_preprocessor("linear")
    pre.fit(select_features(df.iloc[train_idx]))
    names = list(pre.get_feature_names_out())
    assert len(names) == 1 + 1 + 3 + 3 + 3  # history, recency, binaries, zip, channel
    new = select_features(df.iloc[:5]).assign(zip_code="Mars")
    out = pre.transform(new)
    zip_cols = [i for i, n in enumerate(names) if "zip_code" in n]
    assert (out[:, zip_cols] == 0).all()  # unseen level -> all zeros, no error


def test_tree_preprocessor_layout_and_unseen_category():
    df = _frame()
    split = make_split(df)
    d = prepare_fold(df, split, 0, "tree")
    idx = tree_categorical_indices()
    names = list(d.preprocessor.get_feature_names_out())
    assert [names[i].split("__")[1] for i in idx] == ["zip_code", "channel"]
    assert set(np.unique(d.X_train[:, idx[0]])) <= {0, 1, 2}
    out = d.preprocessor.transform(select_features(df.iloc[:5]).assign(channel="Fax"))
    assert (out[:, idx[1]] == -1).all()


def test_prepare_dev_test_fits_on_dev_only():
    df = _frame()
    split = make_split(df)
    d = prepare_dev_test(df, split, "linear")
    assert len(d.X_train) == len(split.dev_idx)
    assert len(d.X_eval) == len(split.test_idx)
    scaler = d.preprocessor.named_transformers_["recency"]
    assert scaler.mean_[0] == pytest.approx(df.iloc[split.dev_idx]["recency"].mean())


def test_invalid_preprocessor_kind():
    with pytest.raises(ValueError):
        make_preprocessor("deep")
