import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MONKEYS = ["monkeyF", "monkeyN"]
ROIS = ["V1", "V4", "IT"]

AREA_CHANNELS = {
    "monkeyF": {"V1": (0, 512), "IT": (512, 832), "V4": (832, 1024)},
    "monkeyN": {"V1": (0, 512), "V4": (512, 768), "IT": (768, 1024)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw TVSD monkey ephys data into NumPy arrays and generate a metadata CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../data/raw/tvsd"),
        help="Directory containing the downloaded raw .mat files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/processed"),
        help="Directory to save the metadata CSV file.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="things_metadata.csv",
        help="Name of the output metadata CSV file.",
    )
    parser.add_argument(
        "--monkey",
        type=str,
        choices=["monkeyF", "monkeyN", "both"],
        default="both",
        help="Select which monkey to process.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        choices=["V1", "V4", "IT", "all"],
        default="all",
        help="Select which ROI to process.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="If set, slices the dataset to use only the first image per category.",
    )
    return parser.parse_args()


def decode_hdf5_strings(f: h5py.File, dataset_reference: str) -> list:
    references = np.array(f[dataset_reference]).ravel()
    decoded_strings = []

    for ref in references:
        char_array = np.array(f[ref]).ravel()
        string_value = "".join(chr(c) for c in char_array)
        decoded_strings.append(string_value.replace("\\", "/"))

    return decoded_strings


def load_image_metadata(mat_file_path: Path) -> list:
    if not mat_file_path.exists():
        raise FileNotFoundError(
            f"Could not locate metadata at {mat_file_path}. Run 0.2.download_monkey_responses.py first."
        )

    print(f"Reading HDF5 structured image data from {mat_file_path}...")
    with h5py.File(mat_file_path, "r") as f:
        train_paths = decode_hdf5_strings(f, "train_imgs/things_path")

    return train_paths


def filter_trial_indices(train_paths: list, use_subset: bool) -> list:
    kept_indices = []
    seen_categories = set()

    for idx, path in enumerate(train_paths):
        category = path.split("/")[0]
        if use_subset:
            if category not in seen_categories:
                seen_categories.add(category)
                kept_indices.append(idx)
        else:
            kept_indices.append(idx)

    return kept_indices


def process_monkey_rois(
    input_dir: Path,
    responses_dir: Path,
    monkeys: list,
    rois: list,
    train_paths: list,
    path_to_id: dict,
    kept_indices: list,
) -> list:
    metadata_rows = []
    row_id_counter = 0

    print(f"Processing electrophysiology matrices and saving to {responses_dir}")
    for monkey in monkeys:
        mua_file_path = input_dir / monkey / "THINGS_normMUA.mat"
        if not mua_file_path.exists():
            raise FileNotFoundError(f"Could not locate MUA file for {monkey} at {mua_file_path}.")

        with h5py.File(mua_file_path, "r") as f_mua:
            raw_train_mua = np.array(f_mua["train_MUA"])

            for roi in rois:
                lo, hi = AREA_CHANNELS[monkey][roi]

                roi_features = raw_train_mua[kept_indices, lo:hi]

                npy_filename = f"{monkey}_{roi}_train_responses.npy"
                npy_save_path = responses_dir / npy_filename

                np.save(npy_save_path, roi_features)
                print(f"Saved {npy_filename} | Shape: {roi_features.shape}")

                for new_idx, orig_idx in enumerate(kept_indices):
                    img_path = train_paths[orig_idx]
                    image_id = path_to_id[img_path]
                    category = img_path.split("/")[0]

                    metadata_rows.append(
                        {
                            "row_id": row_id_counter,
                            "monkey": monkey,
                            "image_id": image_id,
                            "category": category,
                            "ROI": roi,
                            "image_path": img_path,
                            "response_file_name": f"{responses_dir.name}/{npy_filename}",
                            "response_file_index": new_idx,
                        }
                    )
                    row_id_counter += 1

    return metadata_rows


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = args.output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    selected_monkeys = MONKEYS if args.monkey == "both" else [args.monkey]
    selected_rois = ROIS if args.roi == "all" else [args.roi]

    mat_file_path = args.input_dir / "monkeyF" / "_logs" / "things_imgs.mat"
    train_paths = load_image_metadata(mat_file_path)

    kept_indices = filter_trial_indices(train_paths, args.subset)

    kept_paths = [train_paths[i] for i in kept_indices]
    unique_kept_paths = sorted(list(set(kept_paths)))
    path_to_id = {img_path: idx for idx, img_path in enumerate(unique_kept_paths)}
    print(f"Total unique images: {len(unique_kept_paths)} (subset mode: {args.subset})")

    metadata_rows = process_monkey_rois(
        input_dir=args.input_dir,
        responses_dir=responses_dir,
        monkeys=selected_monkeys,
        rois=selected_rois,
        train_paths=train_paths,
        path_to_id=path_to_id,
        kept_indices=kept_indices,
    )

    csv_path = args.output_dir / args.csv_name
    df = pd.DataFrame(metadata_rows)
    df.to_csv(csv_path, index=False)
    print(f"\nMetadata CSV written with {len(df)} entries at: {csv_path}")


if __name__ == "__main__":
    main()
