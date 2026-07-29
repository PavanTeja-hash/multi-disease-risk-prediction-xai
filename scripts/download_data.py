"""
Download both raw datasets so the project can be reproduced from a fresh clone.

The raw CSVs are not committed to the repository (they are large and freely
available), so this script fetches them on demand.

Usage:
    python scripts/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

# The cardiovascular dataset is distributed via Kaggle, which requires a login.
# This GitHub mirror serves the identical file (verified by SHA-256 against a
# second independent mirror) so the download needs no credentials.
HEART_URL = (
    "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv"
)
HEART_SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_diabetes() -> None:
    """Fetch the CDC Diabetes Health Indicators dataset from the UCI repository."""
    if config.DIABETES_RAW.exists():
        print(f"[skip] {config.DIABETES_RAW.name} already present")
        return

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise SystemExit(
            "ucimlrepo is required to download the diabetes dataset.\n"
            "Install it with:  pip install ucimlrepo"
        )

    import pandas as pd

    print("[download] CDC Diabetes Health Indicators (UCI id=891)...")
    dataset = fetch_ucirepo(id=891)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    frame.to_csv(config.DIABETES_RAW, index=False)
    print(f"[saved] {config.DIABETES_RAW.name}  shape={frame.shape}")


def download_heart() -> None:
    """Fetch the Cardiovascular Disease dataset and verify its checksum."""
    if config.HEART_RAW.exists():
        print(f"[skip] {config.HEART_RAW.name} already present")
        return

    print("[download] Cardiovascular Disease dataset...")
    urllib.request.urlretrieve(HEART_URL, config.HEART_RAW)

    actual = _sha256(config.HEART_RAW)
    if actual != HEART_SHA256:
        config.HEART_RAW.unlink()
        raise SystemExit(
            "Checksum mismatch for cardio_train.csv - refusing to use the file.\n"
            f"  expected {HEART_SHA256}\n  actual   {actual}"
        )
    print(f"[saved] {config.HEART_RAW.name}  checksum verified")


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.ensure_directories()
    download_diabetes()
    download_heart()
    print("\nBoth datasets are ready in data/raw/.")


if __name__ == "__main__":
    main()
