"""Download the Diabetes 130-US hospitals dataset.

Two sources are supported:
  1. UCI ML Repository (primary, direct HTTP)
  2. Kaggle (`mramirasgari/diabetes-130-us-hospitals-readmission`)

If the UCI URL fails (it has changed several times), the Kaggle fallback
is used automatically. The result is always the same two CSV files in
`data/raw/`: `diabetic_data.csv` and `IDS_mapping.csv`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

# --- Sources ---------------------------------------------------------------

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130+us+hospitals+for+years+1999+2008.zip"
)

KAGGLE_SLUG = "mramirasgari/diabetes-130-us-hospitals-readmission"

ARCHIVE_NAME = "diabetes_130_us_hospitals.zip"
EXPECTED_CSVS = ("diabetic_data.csv", "IDS_mapping.csv")
CHUNK_SIZE = 1 << 20  # 1 MiB


# --- Core helpers ----------------------------------------------------------


def compute_sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file, streaming in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_present(dest_dir: Path) -> bool:
    return all((dest_dir / name).is_file() for name in EXPECTED_CSVS)


def _extract_csvs(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Extract the two expected CSVs from a zip, flattening nested paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in EXPECTED_CSVS:
                target = dest_dir / name
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                logger.info("Extracted %s -> %s", member, target.name)
                extracted.append(target)

    missing = set(EXPECTED_CSVS) - {p.name for p in extracted}
    if missing:
        raise FileNotFoundError(
            f"Archive {archive_path.name} is missing expected CSVs: {sorted(missing)}"
        )
    return extracted


# --- Source 1: UCI ---------------------------------------------------------


def download_from_uci(url: str, dest_dir: Path) -> Path:
    """Download the UCI zip. Returns the archive path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ARCHIVE_NAME
    logger.info("Trying UCI: %s", url)
    urlretrieve(url, archive_path)
    return archive_path


# --- Source 2: Kaggle ------------------------------------------------------


def download_from_kaggle(slug: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset as a zip via the Kaggle CLI."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Trying Kaggle: %s", slug)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                slug,
                "-p",
                str(dest_dir),
                "--force",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`kaggle` CLI not found. Install it with `uv tool install kaggle`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Kaggle download failed for {slug}: {exc.stderr.strip()}") from exc

    # Kaggle names the zip after the slug's last segment.
    zip_name = slug.split("/")[-1] + ".zip"
    archive_path = dest_dir / zip_name
    if not archive_path.is_file():
        raise FileNotFoundError(f"Expected Kaggle archive not found: {archive_path}")
    return archive_path


# --- Orchestration ---------------------------------------------------------


def fetch(url: str = UCI_URL, slug: str = KAGGLE_SLUG, dest_dir: Path = RAW_DIR) -> list[Path]:
    """Fetch the dataset, trying UCI first then Kaggle. Returns CSV paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _already_present(dest_dir):
        logger.info("CSVs already present in %s — skipping download", dest_dir)
        return [dest_dir / n for n in EXPECTED_CSVS]

    archive: Path | None = None

    # Try UCI
    try:
        archive = download_from_uci(url, dest_dir)
    except Exception as exc:
        logger.warning("UCI download failed (%s); falling back to Kaggle", exc)
        archive = None

    # Fallback to Kaggle
    if archive is None:
        archive = download_from_kaggle(slug, dest_dir)

    # Extract
    csvs = _extract_csvs(archive, dest_dir)

    # Clean up archive
    try:
        archive.unlink()
        logger.info("Removed archive: %s", archive.name)
    except OSError:
        pass

    return csvs


# --- CLI -------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Download the UCI Diabetes 130-US hospitals dataset",
    )
    parser.add_argument("--url", default=UCI_URL, help="Override the UCI URL")
    parser.add_argument("--slug", default=KAGGLE_SLUG, help="Override the Kaggle slug")
    args = parser.parse_args()

    csvs = fetch(url=args.url, slug=args.slug, dest_dir=RAW_DIR)

    print()
    print("Files in data/raw/:")
    for csv in csvs:
        size_mb = csv.stat().st_size / (1024 * 1024)
        digest = compute_sha256(csv)
        print(f"  {csv.name:25s}  {size_mb:7.2f} MB  sha256={digest[:16]}...")

    print()
    print("✅ Download + extraction complete.")


if __name__ == "__main__":
    main()
