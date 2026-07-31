import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from diffusion_brain_alignment.metrics import calc_rdm_matrix, compute_rsa_score


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare Monkeys Representations using RSA (Noise Ceiling)"
    )

    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects to include in the comparison (e.g. monkeyF monkeyN).",
    )

    parser.add_argument(
        "--rois",
        nargs="+",
        type=str,
        default=["V1", "V4", "IT"],
        help="List of ROIs to process (e.g. IT V1 V4)",
    )

    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("../data/processed/things_metadata.csv"),
        help="Path to the generated metadata CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("../data/results/monkey_rsa_comparison.csv"),
        help="Path to the output CSV file to save results.",
    )

    args = parser.parse_args()
    return args


def get_averaged_representations(df, monkey, roi, metadata_csv_path):
    subset = df[(df["monkey"] == monkey) & (df["ROI"] == roi)]
    if subset.empty:
        return None

    response_file_relative = subset.iloc[0]["response_file_name"]
    np_file_path = metadata_csv_path.parent / response_file_relative

    if not np_file_path.exists():
        return None

    raw_representations = np.load(np_file_path)

    image_ids = subset["image_id"].values
    indices = subset["response_file_index"].values.astype(int)

    reps = raw_representations[indices]

    df_reps = pd.DataFrame(reps)
    df_reps["image_id"] = image_ids
    grouped_reps = df_reps.groupby("image_id").mean()

    return grouped_reps


def process_combination(monkey1, monkey2, roi, df, args, device):
    reps1 = get_averaged_representations(df, monkey1, roi, args.metadata_csv)
    reps2 = get_averaged_representations(df, monkey2, roi, args.metadata_csv)

    if reps1 is None or reps2 is None:
        return None

    common_ids = np.intersect1d(reps1.index, reps2.index)
    if len(common_ids) < 3:
        return None

    aligned_reps1 = reps1.loc[common_ids].values
    aligned_reps2 = reps2.loc[common_ids].values

    tensor1 = torch.tensor(aligned_reps1, dtype=torch.float32, device=device)
    tensor2 = torch.tensor(aligned_reps2, dtype=torch.float32, device=device)

    with torch.no_grad():
        rdm1 = calc_rdm_matrix(tensor1)
        rdm2 = calc_rdm_matrix(tensor2)
        score = compute_rsa_score(rdm1, rdm2).item()

    return {
        "monkey1": monkey1,
        "monkey2": monkey2,
        "ROI": roi,
        "common_images": len(common_ids),
        "rsa_score": score,
    }


def main():
    args = parse_arguments()

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {args.metadata_csv}")

    df = pd.read_csv(args.metadata_csv)

    available_monkeys = set(df["monkey"].unique())
    requested_monkeys = [m for m in args.monkeys if m in available_monkeys]
    unique_monkeys = sorted(list(set(requested_monkeys)))

    if len(unique_monkeys) < 2:
        raise ValueError(
            f"Need at least 2 valid monkeys to perform comparison. Valid found: {unique_monkeys}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Compute Device: {device}\n")

    monkey_pairs = list(itertools.combinations(unique_monkeys, 2))
    combinations = list(itertools.product(monkey_pairs, args.rois))
    progress_bar = tqdm(combinations, desc="Computing Noise Ceilings...", unit="comb")
    all_results = []

    for (monkey1, monkey2), roi in progress_bar:
        progress_bar.set_description(f"[{monkey1} vs {monkey2} | {roi}]")

        result_dict = process_combination(monkey1, monkey2, roi, df, args, device)

        if result_dict:
            all_results.append(result_dict)
            progress_bar.set_postfix(score=f"{result_dict['rsa_score']:.4f}")
        else:
            progress_bar.set_postfix(status="Skipped (Insufficient Data)")

    if all_results:
        print("\nSaving results to disk...")
        results_df = pd.DataFrame(all_results)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        print(
            f"Successfully processed {len(all_results)} comparisons. Report saved to: {args.output_csv}"
        )
    else:
        print("\nNo valid RSA comparisons could be completed.")


if __name__ == "__main__":
    main()
