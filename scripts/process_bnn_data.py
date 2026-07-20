import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed"
TSVD_DATA_DIR = RAW_DATA_DIR / "TVSD"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "things_metadata.csv"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

ROIS = ["V1", "V4", "IT"]
MONKEYS = ["monkeyF", "monkeyN"]

area_channels = {
    "monkeyF": {"V1": (0, 512), "IT": (512, 832), "V4": (832, 1024)},
    "monkeyN": {"V1": (0, 512), "V4": (512, 768), "IT": (768, 1024)},
}

def decode_hdf5_strings(f, dataset_reference):
    references = np.array(f[dataset_reference]).ravel()
    decoded_strings = []
    
    for ref in references:
        char_array = np.array(f[ref]).ravel()
        string_value = "".join(chr(c) for c in char_array)
        decoded_strings.append(string_value.replace("\\", "/"))
        
    return decoded_strings


def generate_metadata_and_arrays(dataset_path=TSVD_DATA_DIR, output_csv=OUTPUT_CSV_PATH):
    mat_file_path = Path(dataset_path) / "monkeyF" / "_logs" / "things_imgs.mat"
    
    if not mat_file_path.exists():
        raise FileNotFoundError(
            f"Could not locate the metadata file at: {mat_file_path}. "
            "Make sure you download the data before running this script."
        )

    print(f"Reading HDF5 structured image data from {mat_file_path}...")
    with h5py.File(mat_file_path, "r") as f:
        train_paths = decode_hdf5_strings(f, "train_imgs/things_path")

    unique_paths = sorted(list(set(train_paths)))
    path_to_id = {img_path: idx for idx, img_path in enumerate(unique_paths)}
    print(f"Found {len(unique_paths)} unique images.")

    metadata_rows = []
    row_id_counter = 0

    print("Processing electrophysiology matrix files and writing .npz maps...")
    
    for monkey in MONKEYS:
        mua_file_path = Path(dataset_path) / monkey / "THINGS_normMUA.mat"
        if not mua_file_path.exists():
            raise FileNotFoundError(
                f"Could not locate the MUA file for {monkey} at: {mua_file_path}. "
                "Make sure you download the data before running this script."
            )
            
        with h5py.File(mua_file_path, "r") as f_mua:
            raw_train_mua = np.array(f_mua["train_MUA"])
            if raw_train_mua.shape[0] != len(train_paths):
                raw_train_mua = raw_train_mua.T

            for roi in ROIS:
                lo, hi = area_channels[monkey][roi]
                roi_features = raw_train_mua[:, lo:hi]
                
                npz_filename = f"{monkey}_{roi}_train_responses.npz"
                npz_save_path = PROCESSED_DATA_DIR / npz_filename
                
                np.savez_compressed(npz_save_path, representations=roi_features)
                print(f"Created file {npz_filename} with shape {roi_features.shape}")
                
                for trial_idx, img_path in enumerate(train_paths):
                    image_id = path_to_id[img_path]
                    category = img_path.split("/")[0]
                    
                    metadata_rows.append({
                        "row_id": row_id_counter,
                        "monkey": monkey,
                        "image_id": image_id,
                        "category": category,
                        "ROI": roi,
                        "image_path": img_path,
                        "response_file_name": npz_filename,
                        "response_file_index": trial_idx
                    })
                    row_id_counter += 1

    df = pd.DataFrame(metadata_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nCSV written with {len(df)} entries at: {output_csv}")

if __name__ == "__main__":
    generate_metadata_and_arrays()
