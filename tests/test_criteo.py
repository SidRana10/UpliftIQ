import dataclasses
import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import upliftiq.data.criteo as cr
import upliftiq.data.download as dl
from upliftiq.data.criteo import (
    assert_no_leakage,
    select_features,
    summarize_criteo,
    validate_criteo,
    validation_errors,
)
from upliftiq.data.registry import CRITEO, HILLSTROM


def make_frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=list(CRITEO.features))
    df["treatment"] = (rng.random(n) < 0.85).astype(int)
    df["exposure"] = df["treatment"] * (rng.random(n) < 0.03).astype(int)
    df["visit"] = (rng.random(n) < 0.05).astype(int)
    df["conversion"] = df["visit"] * (rng.random(n) < 0.06).astype(int)
    return df[list(CRITEO.columns)]


def write_gz(df, path):
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    return path


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def gz(tmp_path, frame):
    return write_gz(frame, tmp_path / "c.csv.gz")


def test_spec_and_hillstrom_unchanged():
    assert CRITEO.n_rows == 13_979_592 and len(CRITEO.features) == 12
    assert CRITEO.columns[-4:] == ("treatment", "exposure", "conversion", "visit")
    assert HILLSTROM.n_rows == 64_000 and HILLSTROM.sha256.startswith("0e589332")


def test_summary_matches_pandas_and_is_chunk_invariant(gz, frame):
    a = summarize_criteo(gz, chunksize=700)
    b = summarize_criteo(gz, chunksize=100_000)
    assert a["n_rows"] == b["n_rows"] == len(frame)
    for f in CRITEO.features:
        assert a["features"][f]["mean"] == pytest.approx(frame[f].mean(), abs=1e-9)
        assert a["features"][f]["std"] == pytest.approx(frame[f].std(ddof=0), abs=1e-9)
        assert (
            a["features"][f]["min"] == frame[f].min() and a["features"][f]["max"] == frame[f].max()
        )
        assert a["features"][f]["mean"] == pytest.approx(b["features"][f]["mean"], abs=1e-9)
    t = frame["treatment"] == 1
    assert a["arms"]["treated"]["n"] == t.sum() and a["arms"]["control"]["n"] == (~t).sum()
    assert a["arms"]["treated"]["share"] == pytest.approx(t.mean())
    assert a["arms"]["treated"]["visit_rate"] == pytest.approx(frame.loc[t, "visit"].mean())
    assert a["arms"]["control"]["conversion_rate"] == pytest.approx(
        frame.loc[~t, "conversion"].mean()
    )
    assert a["overall"]["visit_rate"] == pytest.approx(frame["visit"].mean())
    assert a["arms"]["control"]["exposed_share"] == 0
    assert sum(r["n"] for r in a["treatment_by_exposure"]) == len(frame)


def test_valid_file_passes(gz, frame):
    s = summarize_criteo(gz, chunksize=1000)
    assert validation_errors(s, expected_rows=len(frame)) == []


def test_wrong_row_count_missing_values_and_bad_binaries_detected(tmp_path, frame):
    bad = frame.copy()
    bad.loc[0, "f3"] = np.nan
    bad.loc[1, "visit"] = 2
    s = summarize_criteo(write_gz(bad, tmp_path / "b.csv.gz"))
    errs = " | ".join(validation_errors(s, expected_rows=len(frame) + 1))
    assert (
        "expected" in errs
        and "missing values in f3" in errs
        and "non-binary values in visit" in errs
    )
    with pytest.raises(ValueError):
        validate_criteo(write_gz(frame, tmp_path / "g.csv.gz"))  # default expects 13,979,592 rows


def test_missing_column_and_exposure_without_treatment_detected(tmp_path, frame):
    s = summarize_criteo(write_gz(frame.drop(columns=["f11"]), tmp_path / "c.csv.gz"))
    assert any("unexpected columns" in e for e in validation_errors(s, len(frame)))
    bad = frame.copy()
    bad.loc[bad["treatment"] == 0, "exposure"] = 1
    s2 = summarize_criteo(write_gz(bad, tmp_path / "d.csv.gz"))
    assert s2["exposed_without_treatment"] > 0
    assert any("exposed" in e for e in validation_errors(s2, len(frame)))


def test_conversion_without_visit_is_reported_not_an_error(tmp_path, frame):
    odd = frame.copy()
    odd.loc[0, ["visit", "conversion"]] = [0, 1]
    s = summarize_criteo(write_gz(odd, tmp_path / "o.csv.gz"))
    assert s["conversion_without_visit"] == 1
    assert validation_errors(s, len(frame)) == []


def test_exposure_and_treatment_never_enter_features(frame):
    assert list(select_features(frame).columns) == list(CRITEO.features)
    for bad in ("exposure", "treatment", "visit", "conversion"):
        with pytest.raises(ValueError):
            assert_no_leakage([*CRITEO.features, bad])
    with pytest.raises(ValueError):
        assert_no_leakage(list(CRITEO.features)[:-1])


def test_main_writes_small_summary_files(gz, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(cr, "validation_errors", lambda s: [])
    assert cr.main(["--path", str(gz), "--out", str(out), "--chunksize", "1000"]) == 0
    data = json.loads((out / "dataset_summary.json").read_text())
    assert data["n_rows"] == 5000
    md = (out / "dataset_summary.md").read_text()
    assert "exposure" in md and "PASSED" in md


# ---------- download ----------


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src.gz"
    src.write_bytes(gzip.compress(b"a,b\n1,2\n"))
    return src


@pytest.fixture
def unpinned(monkeypatch):
    """Registry without the pinned checksum, so tiny synthetic files can be 'downloaded'."""
    monkeypatch.setattr(dl, "CRITEO", dataclasses.replace(CRITEO, sha256=None))


def test_download_streams_hashes_and_records_provenance(unpinned, tmp_path, source):
    import hashlib

    dest = tmp_path / "raw"
    path = dl.download_criteo(dest, url=source.as_uri())
    assert path.name == CRITEO.filename
    rec = json.loads((dest / "criteo-uplift-v2.1.source.json").read_text())
    assert rec["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert rec["official_source"] is False and rec["size_bytes"] == source.stat().st_size
    with pytest.raises(FileExistsError):
        dl.download_criteo(dest, url=source.as_uri())
    assert dl.download_criteo(dest, url=source.as_uri(), force=True).exists()


def test_download_checksum_mismatch_removes_partial_file(tmp_path, source):
    dest = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        dl.download_criteo(dest, url=source.as_uri(), expected_sha256="0" * 64)
    assert list(dest.iterdir()) == []


def test_download_enforces_the_pinned_registry_checksum_by_default(tmp_path, source):
    assert CRITEO.sha256 == "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        dl.download_criteo(tmp_path / "raw", url=source.as_uri())


def test_default_download_source_is_the_publisher_hosted_hugging_face_copy():
    assert CRITEO.hf_repo == "criteo/criteo-uplift"
    assert CRITEO.download_url.startswith("https://huggingface.co/datasets/criteo/criteo-uplift/")
    assert CRITEO.download_url.endswith(CRITEO.official_url.rsplit("/", 1)[1])


def test_download_checks_disk_space_first(unpinned, tmp_path, source, monkeypatch):
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: usage._replace(free=1024))
    with pytest.raises(RuntimeError, match="free"):
        dl.download_criteo(tmp_path / "raw", url=source.as_uri())
    assert not (tmp_path / "raw" / CRITEO.filename).exists()


def test_cli_defaults_to_hillstrom_and_criteo_flag(unpinned, tmp_path, source, monkeypatch):
    calls = {}
    monkeypatch.setattr(dl, "download_hillstrom", lambda d, f: calls.setdefault("h", d) and d)
    assert dl.main(["--dest", str(tmp_path)]) == 0 and "h" in calls
    assert (
        dl.main(["--dataset", "criteo", "--dest", str(tmp_path / "c"), "--url", source.as_uri()])
        == 0
    )
    assert (tmp_path / "c" / CRITEO.filename).exists()


@pytest.mark.network
def test_real_criteo_file_if_downloaded():
    from upliftiq.data.registry import RAW_DIR

    path = RAW_DIR / CRITEO.filename
    if not path.exists():
        pytest.skip("Criteo not downloaded")
    assert validation_errors(summarize_criteo(path)) == []


def test_dataclass_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CRITEO.n_rows = 1
