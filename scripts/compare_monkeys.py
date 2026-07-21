import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rsatoolbox

BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed"
RESULTS_DATA_DIR = BASE_DATA_DIR / "results"
CSV_METADATA_PATH = PROCESSED_DATA_DIR / "things_metadata_subset.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare monkeys Representations using RSA")
    
    parser.add_argument("--rdm_metric", type=str, default="correlation", 
                        help="Distance metric for computing RDMs")
    parser.add_argument("--compare_method", type=str, default="rho-a", 
                        help="Metric for comparing RDMs")
    default_output = str(RESULTS_DATA_DIR / "monkey_rsa_comparison.csv")
    parser.add_argument("--output_path", type=str, default=default_output, 
                        help="Full path to save the output CSV file")
    return parser.parse_args()


def get_averaged_representations(df, monkey, roi, processed_dir):
    # This functions averages across responses for the same image
    # This isn't needed now but will be helpful if we add the 100 test images
    subset = df[(df["monkey"] == monkey) & (df["ROI"] == roi)]
    if subset.empty:
        return None, None
    
    npz_filename = subset.iloc[0]["response_file_name"]
    npz_file_path = processed_dir / npz_filename
    
    if not npz_file_path.exists():
        print(f"Warning: Data file {npz_file_path} not found.")
        return None, None
        
    raw_representations = np.load(npz_file_path)["representations"]
    
    unique_ids = []
    averaged_reps = []
    
    for img_id, group in subset.groupby("image_id"):
        indices = group["response_file_index"].values
        avg_rep = raw_representations[indices].mean(axis=0)
        unique_ids.append(img_id)
        averaged_reps.append(avg_rep)
        
    return np.array(unique_ids), np.array(averaged_reps)



def main():
    args = parse_arguments()
    
    if not CSV_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {CSV_METADATA_PATH}")
        
    df = pd.read_csv(CSV_METADATA_PATH)
    
    monkeys = df["monkey"].unique()
    rois = df["ROI"].unique()
    
    if len(monkeys) < 2:
        raise ValueError(f"Need at least 2 monkeys in the dataset to perform comparison. Found: {len(monkeys)}")
        
    monkey1, monkey2 = monkeys[0], monkeys[1]
    
    results = []
    
    print(f"Starting RSA Comparison: {monkey1} vs {monkey2}")
    print(f"RDM Creation Metric: '{args.rdm_metric}' | RDM Comparison Method: '{args.compare_method}'\n")

    for roi in rois:
        print(f"Processing ROI: {roi}...")
        
        ids1, reps1 = get_averaged_representations(df, monkey1, roi, PROCESSED_DATA_DIR)
        ids2, reps2 = get_averaged_representations(df, monkey2, roi, PROCESSED_DATA_DIR)
        
        if ids1 is None or ids2 is None:
            print(f"Insufficient data for ROI {roi}. Skipping.\n")
            continue
            
        common_ids = np.intersect1d(ids1, ids2)
        common_ids = common_ids[:10000]
        print(f"Found {len(common_ids)} overlapping unique images.")
        
        if len(common_ids) < 3:
            print("Not enough common images to compute RSA. Skipping.\n")
            continue
            
        aligned_reps1 = np.array([reps1[np.where(ids1 == cid)[0][0]] for cid in common_ids])
        aligned_reps2 = np.array([reps2[np.where(ids2 == cid)[0][0]] for cid in common_ids])
        
        dataset1 = rsatoolbox.data.Dataset(aligned_reps1)
        dataset2 = rsatoolbox.data.Dataset(aligned_reps2)
        
        try:
            rdm1 = rsatoolbox.rdm.calc_rdm(dataset1, method=args.rdm_metric)
            rdm2 = rsatoolbox.rdm.calc_rdm(dataset2, method=args.rdm_metric)
        except Exception as e:
            print(f"RDM calculation failed: {e}")
            continue
            
        try:
            score = rsatoolbox.rdm.compare(rdm1, rdm2, method=args.compare_method)
            similarity_value = float(score[0][0])
            print(f"Alignment Score: {similarity_value:.4f}\n")
        except Exception as e:
            print(f"RDM comparison failed: {e}")
            continue
            
        results.append({
            "monkey1": monkey1,
            "monkey2": monkey2,
            "ROI": roi,
            "common_images": len(common_ids),
            "rdm_metric": args.rdm_metric,
            "compare_method": args.compare_method,
            "rsa_score": similarity_value
        })
        
    if results:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"RSA comparison report saved to: {output_path}")
    else:
        print("No valid RSA comparisons could be completed.")

if __name__ == "__main__":
    main()