import importlib

import pytest

import upliftiq
from upliftiq.data.registry import PROJECT_ROOT


def test_version():
    assert upliftiq.__version__


@pytest.mark.parametrize("sub", ["data", "modeling", "evaluation", "backend"])
def test_subpackages_importable(sub):
    importlib.import_module(f"upliftiq.{sub}")


def test_project_layout():
    for p in ("README.md", "pyproject.toml", ".gitignore", "frontend", "docs/datasets.md"):
        assert (PROJECT_ROOT / p).exists(), p


def test_gitignore_excludes_raw_data_and_envs():
    text = (PROJECT_ROOT / ".gitignore").read_text()
    assert ".venv/" in text
    assert "data/raw/*" in text
    assert "*.csv" in text
