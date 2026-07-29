import argparse
from pathlib import Path
import urllib.request
import zipfile

from tqdm.auto import tqdm

DEFAULT_IMAGE_URL = "https://osf.io/download/rdxy2/"
DEFAULT_ZIP_NAME = "images_THINGS.zip"
DEFAULT_OUTPUT_DIR = Path("../data/raw/images")
IMAGES_PASSWORD = b"things4all"


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract the THINGS image dataset.")
    parser.add_argument(
        "--url", type=str, default=DEFAULT_IMAGE_URL, help="URL of the images dataset archive."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the extracted images.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip file after extraction.",
    )
    return parser.parse_args()


def download_file(url: str, dest_path: Path):
    print(f"Downloading from {url}...")

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(dest_path, "wb") as file,
            tqdm(
                desc=dest_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            while True:
                chunk = response.read(8 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                progress_bar.update(len(chunk))

    print("\nDownload complete.")


def extract_archive(archive_path: Path, extract_to: Path):
    print(f"Extracting {archive_path.name} to {extract_to} (this may take a while)...")

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(path=extract_to, pwd=IMAGES_PASSWORD)

    print("Extraction complete.")


def main():
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir.parent / DEFAULT_ZIP_NAME

    if not archive_path.exists() and not (output_dir / "object_images").exists():
        download_file(args.url, archive_path)
    else:
        print("Archive or extracted folder already exists. Skipping download.")

    if archive_path.exists() and not (output_dir / "object_images").exists():
        extract_archive(archive_path, output_dir)
    else:
        print("Files already appear to be extracted. Skipping extraction.")

    if not args.keep_archive and archive_path.exists():
        print(f"Cleaning up archive {archive_path.name}...")
        archive_path.unlink()

    print(f"Success! Image data is ready in: {output_dir}")


if __name__ == "__main__":
    main()
