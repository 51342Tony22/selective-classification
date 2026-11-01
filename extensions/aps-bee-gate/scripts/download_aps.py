from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


URL = "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
REQUIRED_FILES = {
    "aps_failure_training_set.csv",
    "aps_failure_test_set.csv",
    "aps_failure_description.txt",
}


def download_aps(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aps_download_") as temporary:
        archive = Path(temporary) / "aps_failure.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            names = {Path(name).name: name for name in zipped.namelist()}
            missing = sorted(REQUIRED_FILES - set(names))
            if missing:
                raise RuntimeError(f"APS archive missing expected files: {missing}")
            written = []
            for filename in sorted(REQUIRED_FILES):
                destination = output / filename
                with zipped.open(names[filename]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                written.append(destination)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download APS Failure from the UCI repository.")
    parser.add_argument("--output-dir", default="data/raw/aps_failure")
    args = parser.parse_args(argv)
    for path in download_aps(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
