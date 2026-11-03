"""Reproducible download of the original Hillstrom dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from upliftiq.data.registry import CRITEO, HILLSTROM, RAW_DIR


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hillstrom(dest_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download Hillstrom CSV to ``dest_dir`` and verify its checksum.

    Writes a ``hillstrom.source.json`` provenance file next to the CSV.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / HILLSTROM.filename
    if dest.exists() and not force:
        print(f"{dest} already exists; skipping download (use --force to refetch)")
    else:
        tmp = dest.with_suffix(".csv.part")
        print(f"Downloading {HILLSTROM.url}")
        with urllib.request.urlopen(HILLSTROM.url, timeout=60) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.replace(dest)

    digest = sha256_of(dest)
    if digest != HILLSTROM.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {dest}: expected {HILLSTROM.sha256}, got {digest}"
        )
    provenance = {
        "dataset": HILLSTROM.name,
        "source_url": HILLSTROM.url,
        "sha256": digest,
        "size_bytes": dest.stat().st_size,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (dest_dir / "hillstrom.source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dest


MIN_FREE_BYTES = 2 * 1024**3  # compressed Criteo file is ~300 MB; data is streamed, not unpacked


def download_criteo(
    dest_dir: Path = RAW_DIR,
    url: str | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Stream the compressed Criteo v2.1 file to ``dest_dir``, hashing as it downloads.

    ``url`` defaults to the publisher-hosted Hugging Face copy (``criteo/criteo-uplift``); any other
    URL is recorded as an unofficial mirror in
    ``criteo-uplift-v2.1.source.json``. The SHA-256 is checked against ``expected_sha256`` (or the
    registry value) when one is known; otherwise it is only recorded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"only {free / 1024**3:.1f} GiB free in {dest_dir}; need >= 2 GiB")
    dest = dest_dir / CRITEO.filename
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists (use --force to refetch)")
    source = url or CRITEO.download_url
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    print(f"Downloading {source}")
    try:
        with urllib.request.urlopen(source, timeout=60) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        digest = h.hexdigest()
        expected = expected_sha256 or CRITEO.sha256
        if expected and digest != expected:
            raise RuntimeError(f"Checksum mismatch: expected {expected}, got {digest}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    provenance = {
        "dataset": CRITEO.name,
        "source_url": source,
        "official_source": source in (CRITEO.official_url, CRITEO.download_url),
        "source_kind": (
            "publisher-hosted (Hugging Face criteo/criteo-uplift)"
            if source == CRITEO.download_url
            else "official page link"
            if source == CRITEO.official_url
            else "unofficial mirror"
        ),
        "dataset_version": CRITEO.version,
        "original_publisher": "Criteo AI Lab (Diemert et al., AdKDD 2018)",
        "license": CRITEO.license_note,
        "source_page": CRITEO.source_page,
        "sha256": digest,
        "checksum_verified_against": "expected value" if expected else "none (first record)",
        "size_bytes": dest.stat().st_size,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (dest_dir / "criteo-uplift-v2.1.source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a project dataset.")
    parser.add_argument("--dataset", choices=["hillstrom", "criteo"], default="hillstrom")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--url", help="Criteo only: alternative (mirror) URL")
    parser.add_argument("--expected-sha256", help="Criteo only: checksum to enforce")
    args = parser.parse_args(argv)
    if args.dataset == "criteo":
        path = download_criteo(args.dest, args.url, args.expected_sha256, args.force)
    else:
        path = download_hillstrom(args.dest, args.force)
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
