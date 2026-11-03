"""Reproducible download of the original Hillstrom dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from upliftiq.data.registry import HILLSTROM, RAW_DIR


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Hillstrom dataset.")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = download_hillstrom(args.dest, args.force)
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
