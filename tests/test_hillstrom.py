import pandas as pd
import pytest

from upliftiq.data.registry import HILLSTROM, RAW_DIR
from upliftiq.data.validate import validate_hillstrom, validate_hillstrom_frame


def _good_frame(n: int = HILLSTROM.n_rows) -> pd.DataFrame:
    segs = sorted(HILLSTROM.segments)
    return pd.DataFrame(
        {
            "recency": [1] * n,
            "history_segment": ["1) $0 - $100"] * n,
            "history": [50.0] * n,
            "mens": [1] * n,
            "womens": [0] * n,
            "zip_code": ["Urban"] * n,
            "newbie": [0] * n,
            "channel": ["Web"] * n,
            "segment": [segs[i % 3] for i in range(n)],
            "visit": [0] * n,
            "conversion": [0] * n,
            "spend": [0.0] * n,
        }
    )


def test_valid_synthetic_frame_passes():
    assert validate_hillstrom_frame(_good_frame()) == []


def test_wrong_row_count_detected():
    assert any("rows" in e for e in validate_hillstrom_frame(_good_frame(10)))


def test_wrong_columns_detected():
    assert validate_hillstrom_frame(_good_frame().drop(columns=["spend"]))


def test_inconsistent_outcomes_detected():
    df = _good_frame()
    df.loc[0, "spend"] = 5.0
    assert any("spend" in e for e in validate_hillstrom_frame(df))


@pytest.mark.network
def test_real_hillstrom_file():
    path = RAW_DIR / HILLSTROM.filename
    if not path.exists():
        pytest.skip("Hillstrom not downloaded; run python -m upliftiq.data.download")
    from upliftiq.data.download import sha256_of

    assert sha256_of(path) == HILLSTROM.sha256
    df = validate_hillstrom(path)
    assert len(df) == 64_000
    assert df["segment"].value_counts().min() > 20_000
