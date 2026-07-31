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
        description="Compute Bootstrap Confidence Intervals for RSA (GPU Batch Mode)"
    )

    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects (e.g. monkeyF monkeyN)",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        type=str,
        default=["V1", "V4", "IT"],
        help="List of ROIs to process (e.g. IT V1 V4)",
    )
    parser.add_argument(
        "--noise_degrees",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="List of noise degrees (e.g. 0.25 0.5 0.75)",
    )

    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument(
        "--sample_size",
        type=float,
        default=0.8,
        help="Number of images to subsample WITHOUT replacement (if >= 1) or ratio of total data size (if < 1). Defaults to 0.8.",
    )
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("../data/processed/things_metadata.csv"),
        help="Path to the generated metadata CSV file.",
    )
    parser.add_argument(
        "--activations_dir",
        type=Path,
        default=Path("../data/processed/activations"),
        help="Directory containing the extracted ANN .npy features.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("../data/results/rsa_bootstrap_ci_results.csv"),
        help="Path to the output CSV file to append results to.",
    )

    args = parser.parse_args()
    return args


def load_representations(monkey, roi, noise_degree, args, device):
    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {args.metadata_csv}")

    df = pd.read_csv(args.metadata_csv)
    subset_df = df[(df["monkey"] == monkey) & (df["ROI"] == roi)]
    if subset_df.empty:
        raise ValueError(f"No metadata found for {monkey} - {roi}")

    assert subset_df["response_file_name"].nunique() == 1, (
        "Multiple response files found in subset! Expected a single biological array."
    )

    response_file_relative = subset_df.iloc[0]["response_file_name"]
    bio_file_path = args.metadata_csv.parent / response_file_relative

    if not bio_file_path.exists():
        raise FileNotFoundError(f"Biological responses not found at {bio_file_path}")

    bio_features = np.load(bio_file_path)
    response_indices = subset_df["response_file_index"].values.astype(int)
    bio_matrix = bio_features[response_indices]

    noise_rounded = round(noise_degree, 2)
    ai_file_path = args.activations_dir / f"sd15_mid_block_noise_{noise_rounded:.2f}.npy"

    if not ai_file_path.exists():
        raise FileNotFoundError(f"ANN activations not found at {ai_file_path}")

    ai_features = np.load(ai_file_path, mmap_mode="r")
    image_ids = subset_df["image_id"].values.astype(int)
    ai_matrix = ai_features[image_ids]

    ai_tensor = torch.tensor(ai_matrix, dtype=torch.float32, device=device)
    bio_tensor = torch.tensor(bio_matrix, dtype=torch.float32, device=device)

    return ai_tensor, bio_tensor


def bootstrap_rsa_ci_gpu(ai_tensor, bio_tensor, n_bootstrap, ci, sample_size, device, random_seed):
    n_total_images = ai_tensor.shape[0]
    torch.manual_seed(random_seed)

    if sample_size < 1.0:
        actual_sample_size = int(n_total_images * sample_size)
    else:
        actual_sample_size = int(sample_size)

    if actual_sample_size > n_total_images:
        raise ValueError(
            f"Requested sample size ({actual_sample_size}) exceeds available images ({n_total_images})"
        )

    ai_rdm_full = calc_rdm_matrix(ai_tensor)
    bio_rdm_full = calc_rdm_matrix(bio_tensor)

    observed_score = compute_rsa_score(ai_rdm_full, bio_rdm_full).item()

    boot_distribution_tensor = torch.zeros(n_bootstrap, device=device)

    with torch.no_grad():
        for i in range(n_bootstrap):
            idx = torch.randperm(n_total_images, device=device)[:actual_sample_size]
            ai_rdm_boot = ai_rdm_full[idx][:, idx]
            bio_rdm_boot = bio_rdm_full[idx][:, idx]
            score = compute_rsa_score(ai_rdm_boot, bio_rdm_boot)
            boot_distribution_tensor[i] = score

    boot_distribution = boot_distribution_tensor.cpu().numpy()
    alpha = (100 - ci) / 2

    return {
        "score": observed_score,
        "ci_low": float(np.percentile(boot_distribution, alpha)),
        "ci_high": float(np.percentile(boot_distribution, 100 - alpha)),
        "boot_mean": float(np.mean(boot_distribution)),
        "boot_std": float(np.std(boot_distribution, ddof=1)),
        "n_total_images": int(n_total_images),
        "n_subsample": int(actual_sample_size),
        "n_bootstrap": int(n_bootstrap),
        "ci": float(ci),
    }


def process_combination(monkey, roi, noise_degree, args, device):
    """Handles end-to-end processing for a single monkey/ROI/noise combination."""
    try:
        ai_tensor, bio_tensor = load_representations(monkey, roi, noise_degree, args, device)
    except (ValueError, FileNotFoundError):
        return None

    results = bootstrap_rsa_ci_gpu(
        ai_tensor,
        bio_tensor,
        args.n_bootstrap,
        args.ci,
        args.sample_size,
        device,
        args.random_seed,
    )

    results_dict = {
        "monkey": monkey,
        "roi": roi,
        "noise_degree": noise_degree,
        "random_seed": args.random_seed,
        **results,
    }

    return results_dict


def main():
    args = parse_arguments()

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {args.metadata_csv}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Compute Device: {device}\n")

    combinations = list(itertools.product(args.monkeys, args.rois, args.noise_degrees))
    progress_bar = tqdm(combinations, desc="Initializing...", unit="comb")

    all_results = []

    for monkey, roi, noise_degree in progress_bar:
        progress_bar.set_description(f"[{monkey} | {roi} | Noise: {noise_degree}]")

        result_dict = process_combination(monkey, roi, noise_degree, args, device)

        if result_dict:
            all_results.append(result_dict)
            progress_bar.set_postfix(
                score=f"{result_dict['score']:.4f}",
                ci=f"[{result_dict['ci_low']:.4f}, {result_dict['ci_high']:.4f}]",
            )
        else:
            progress_bar.set_postfix(status="Skipped (No Data)")

    if all_results:
        print("\nSaving results to disk...")
        results_df = pd.DataFrame(all_results)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        print(f"Successfully processed {len(all_results)} combinations.")


if __name__ == "__main__":
    main()
