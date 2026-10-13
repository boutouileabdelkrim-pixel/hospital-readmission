"""Unit tests for src.data.download (offline, no network)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from src.data import download


def _make_fake_archive(path: Path, members: dict[str, str]) -> None:
    """Create a zip at path with the given {arcname: content} members."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def test_extract_csvs_flattens_nested_paths(tmp_path: Path) -> None:
    archive = tmp_path / "fake.zip"
    _make_fake_archive(
        archive,
        {
            "dataset_diabetes/diabetic_data.csv": "encounter_id,patient_nbr\n1,100\n",
            "dataset_diabetes/IDS_mapping.csv": "id,description\n1,Emergency\n",
        },
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = download._extract_csvs(archive, out_dir)

    names = sorted(p.name for p in result)
    assert names == ["IDS_mapping.csv", "diabetic_data.csv"]
    assert (out_dir / "diabetic_data.csv").is_file()
    assert (out_dir / "IDS_mapping.csv").is_file()


def test_extract_csvs_raises_when_expected_file_missing(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    _make_fake_archive(
        archive,
        {"dataset_diabetes/diabetic_data.csv": "a\n1\n"},
    )
    with pytest.raises(FileNotFoundError, match=r"IDS_mapping\.csv"):
        download._extract_csvs(archive, tmp_path)


def test_compute_sha256_matches_known_digest(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert download.compute_sha256(f) == expected


def test_already_present_true_when_both_csvs_exist(tmp_path: Path) -> None:
    (tmp_path / "diabetic_data.csv").write_text("x")
    (tmp_path / "IDS_mapping.csv").write_text("y")
    assert download._already_present(tmp_path) is True


def test_already_present_false_when_one_missing(tmp_path: Path) -> None:
    (tmp_path / "diabetic_data.csv").write_text("x")
    assert download._already_present(tmp_path) is False
