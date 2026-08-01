import argparse
from pathlib import Path
import urllib.error
import urllib.request

from tqdm import tqdm

BASE_URL = "https://gin.g-node.org/paolo_papale/TVSD/raw/master/"
DEFAULT_OUTPUT_DIR = Path("../data/raw/tvsd")


def parse_args():
    parser = argparse.ArgumentParser(description="Download Monkey Ephys (TVSD) dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the downloaded neural data.",
    )
    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects to download (e.g., monkeyF monkeyN).",
    )
    return parser.parse_args()


def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name}...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
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
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        print(f"Saved to {dest_path}\n")
    except urllib.error.URLError as e:
        print(f"Failed to download {url}. Error: {e}")


def main():
    args = parse_args()
    output_dir = args.output_dir

    files_to_download = ["monkeyF/_logs/things_imgs.mat"]

    if "monkeyF" in args.monkeys:
        files_to_download.append("monkeyF/THINGS_normMUA.mat")

    if "monkeyN" in args.monkeys:
        files_to_download.append("monkeyN/THINGS_normMUA.mat")

    print(f"Target directory: {output_dir}")

    for file_path in files_to_download:
        dest_path = output_dir / file_path
        url = BASE_URL + file_path

        if dest_path.exists():
            print(f"File already exists: {dest_path}. Skipping download.")
        else:
            download_file(url, dest_path)

    print("Success! Monkey response data is ready.")


if __name__ == "__main__":
    main()
