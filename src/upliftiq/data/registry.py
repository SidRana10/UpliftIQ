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


@dataclass(frozen=True)
class CriteoSpec:
    name: str
    version: str
    source_page: str
    official_url: str
    download_url: str  # publisher-hosted copy (Hugging Face)
    hf_repo: str
    filename: str
    n_rows: int
    features: tuple[str, ...]
    treatment: str
    exposure: str
    outcomes: tuple[str, ...]
    sha256: str | None  # supplied by the project owner for the Hugging Face copy
    license_note: str

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.features, self.treatment, self.exposure, *self.outcomes)


CRITEO = CriteoSpec(
    name="Criteo Uplift Prediction Dataset (corrected / un-biased v2.1)",
    version="v2.1",
    source_page="https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
    official_url="http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
    download_url=(
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
        "criteo-research-uplift-v2.1.csv.gz"
    ),
    hf_repo="criteo/criteo-uplift",
    filename="criteo-uplift-v2.1.csv.gz",
    n_rows=13_979_592,
    features=tuple(f"f{i}" for i in range(12)),
    treatment="treatment",
    exposure="exposure",
    outcomes=("conversion", "visit"),
    sha256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    license_note="CC BY-NC-SA 4.0 (non-commercial, share-alike); cite Diemert et al., AdKDD 2018.",
)
