"""Download and verify UCI Abalone dataset integrity.

Verifies MD5 and SHA-256 checksums to ensure reproducible and tamper-free data pipeline.
Official source: UCI Machine Learning Repository (DOI: 10.24432/C55C7W)
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from typing import Tuple

# Force utf-8 output to prevent Windows cp1252 character encoding crashes
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
FALLBACK_URL = "https://raw.githubusercontent.com/HuyLuan21/abalone-age-estimation/main/data/raw/abalone.data"

EXPECTED_MD5 = "a769fd0119787cac09158fe08971e480"
EXPECTED_SHA256 = "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f"
EXPECTED_SIZE = 191873  # bytes


def calculate_checksums(filepath: str) -> Tuple[str, str, int]:
    """Calculate MD5, SHA-256, and file size for a given file."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    size = 0

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
            size += len(chunk)

    return md5_hash.hexdigest(), sha256_hash.hexdigest(), size


def verify_file(filepath: str) -> bool:
    """Verify file existence and checksum matches."""
    if not os.path.exists(filepath):
        return False
    md5, sha256, size = calculate_checksums(filepath)
    return (
        md5 == EXPECTED_MD5
        and sha256 == EXPECTED_SHA256
        and size == EXPECTED_SIZE
    )


def download_and_verify(
    destination: str = "data/raw/abalone.data",
    force: bool = False
) -> bool:
    """Download the raw Abalone dataset from UCI and verify checksums.

    Args:
        destination: File destination path.
        force: If True, re-download even if a valid file exists.

    Returns:
        bool: True if file exists and passes verification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

    if not force and verify_file(destination):
        md5, sha256, size = calculate_checksums(destination)
        print(f"[OK] File '{destination}' already exists and verified:")
        print(f"     Size:   {size:,} bytes")
        print(f"     MD5:    {md5}")
        print(f"     SHA256: {sha256}")
        return True

    print(f"[*] Downloading dataset to '{destination}'...")
    download_urls = [DATA_URL, FALLBACK_URL]
    downloaded = False

    for url in download_urls:
        try:
            print(f"    Trying {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp, open(destination, "wb") as out_file:
                while chunk := resp.read(8192):
                    out_file.write(chunk)
            downloaded = True
            print("    Download complete.")
            break
        except Exception as e:
            print(f"    Failed ({e}), trying next URL...")

    if not downloaded:
        raise RuntimeError("Failed to download abalone.data from all available sources.")

    md5, sha256, size = calculate_checksums(destination)
    print(f"[*] Downloaded: {size:,} bytes")
    print(f"    Calculated MD5:    {md5}")
    print(f"    Expected MD5:      {EXPECTED_MD5}")
    print(f"    Calculated SHA256: {sha256}")
    print(f"    Expected SHA256:   {EXPECTED_SHA256}")

    if md5 != EXPECTED_MD5 or sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum verification failed! File '{destination}' may be corrupt or modified."
        )

    print("[PASS] Checksum verification passed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify Abalone dataset integrity.")
    parser.add_argument(
        "--dest",
        default="data/raw/abalone.data",
        help="Destination path for abalone.data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if file exists"
    )
    args = parser.parse_args()

    try:
        download_and_verify(args.dest, force=args.force)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
