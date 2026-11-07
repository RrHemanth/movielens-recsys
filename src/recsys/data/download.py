import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://files.grouplens.org/datasets/movielens"


def md5_of_file(path: Path) -> str:
    """Compute the MD5 fingerprint of a file, reading it 1 MB at a time."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fetch_official_md5(dataset: str) -> str:
    """Download the checksum GroupLens publishes next to the zip file."""
    with urllib.request.urlopen(f"{BASE_URL}/{dataset}.zip.md5") as resp:
        text = resp.read().decode()
    match = re.search(r"[0-9a-f]{32}", text)  # an MD5 hash is 32 hex characters
    if match is None:
        raise ValueError(f"Could not find an MD5 hash in: {text!r}")
    return match.group(0)


def download_movielens(raw_dir: Path, dataset: str = "ml-25m") -> Path:
    """Download, verify and unzip a MovieLens dataset. Returns the unzipped folder."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{dataset}.zip"

    # 1. Download (skip if we already have it)
    if zip_path.exists():
        print(f"{zip_path.name} already exists, skipping download.")
    else:
        print(f"Downloading {dataset}.zip (about 250 MB)...")
        temp_path = raw_dir / f"{dataset}.zip.part"
        urllib.request.urlretrieve(f"{BASE_URL}/{dataset}.zip", temp_path)
        temp_path.rename(zip_path)  # only gets the real name once fully downloaded

    # 2. Verify against the official checksum
    expected = fetch_official_md5(dataset)
    actual = md5_of_file(zip_path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch: expected {expected}, got {actual}. "
            f"Delete {zip_path} and run again."
        )
    print(f"Checksum OK: {actual}")

    # 3. Unzip (skip if already unzipped)
    extract_dir = raw_dir / dataset
    if not extract_dir.exists():
        print("Unzipping...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return extract_dir