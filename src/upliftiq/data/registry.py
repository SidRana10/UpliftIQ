"""Static metadata for the datasets used by UpliftIQ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class HillstromSpec:
    name: str
    url: str
    filename: str
    n_rows: int
    columns: tuple[str, ...]
    segments: frozenset[str]
    sha256: str
    license_note: str


HILLSTROM = HillstromSpec(
    name="Hillstrom MineThatData E-Mail Analytics And Data Mining Challenge",
    url=(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    filename="hillstrom.csv",
    n_rows=64_000,
    columns=(
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "segment",
        "visit",
        "conversion",
        "spend",
    ),
    segments=frozenset({"Mens E-Mail", "Womens E-Mail", "No E-Mail"}),
    sha256="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
    license_note=(
        "Published by Kevin Hillstrom (MineThatData) for the 2008 challenge; "
        "freely downloadable, no explicit license stated. Cite the source in any publication."
    ),
)
