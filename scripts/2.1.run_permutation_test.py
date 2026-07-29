import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare Artificial and Biological Representations using RSA"
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
        "--noise_levels",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="List of noise levels (e.g. 0.25 0.5 0.75)",
    )

    parser.add_argument("--rdm_metric", type=str, default="correlation")
    parser.add_argument("--n_permutations", type=int, default=1000)
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
        default=Path("../data/results/rsa_permutation_results.csv"),
        help="Path to the output CSV file to append results to.",
    )

    parser.add_argument(
        "--save_null_dists",
        action="store_true",
        help="Optionally save the permutation null distributions as .npy files for later plotting.",
    )
    parser.add_argument(
        "--null_dists_dir",
        type=Path,
        default=Path("../data/results/null_distributions"),
        help="Directory to save the null distribution .npy arrays.",
    )

    return parser.parse_args()


def load_representations(monkey, roi, noise_level, args):
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

    noise_rounded = round(noise_level, 2)
    ai_file_path = args.activations_dir / f"sd15_mid_block_noise_{noise_rounded:.2f}.npy"

    if not ai_file_path.exists():
        raise FileNotFoundError(f"ANN activations not found at {ai_file_path}")

    ai_features = np.load(ai_file_path, mmap_mode="r")
    image_ids = subset_df["image_id"].values.astype(int)
    ai_matrix = ai_features[image_ids]

    return ai_matrix, bio_matrix


def calc_rdm_correlation_torch(x):
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_norm = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    sim = torch.mm(x_norm, x_norm.t())
    return 1.0 - sim


def generate_rdms(ai_tensor, bio_tensor):
    ai_rdm_tensor = calc_rdm_correlation_torch(ai_tensor)
    bio_rdm_tensor = calc_rdm_correlation_torch(bio_tensor)
    return ai_rdm_tensor, bio_rdm_tensor


def compute_rsa_score(ai_rdm_tensor, bio_rdm_tensor, device):
    num_conditions = bio_rdm_tensor.shape[0]
    i_upper, j_upper = torch.triu_indices(num_conditions, num_conditions, offset=1, device=device)
    ai_vector = ai_rdm_tensor[i_upper, j_upper]
    bio_vector = bio_rdm_tensor[i_upper, j_upper]

    ai_ranks = torch.argsort(torch.argsort(ai_vector)).float()
    bio_ranks = torch.argsort(torch.argsort(bio_vector)).float()

    ai_ranks_centered = ai_ranks - torch.mean(ai_ranks)
    bio_ranks_centered = bio_ranks - torch.mean(bio_ranks)

    n = ai_vector.shape[0]
    expected_variance = ((n**3) - n) / 12.0
    expected_std = math.sqrt(expected_variance)

    ai_ranks_scaled = ai_ranks_centered / expected_std
    bio_ranks_scaled = bio_ranks_centered / expected_std

    correlation = torch.dot(ai_ranks_scaled, bio_ranks_scaled)
    return correlation


def generate_null_distribution(ai_rdm_tensor, bio_rdm_tensor, n_permutations, device, random_seed):
    num_conditions = bio_rdm_tensor.shape[0]

    torch.manual_seed(random_seed)

    null_distribution_tensor = torch.zeros(n_permutations, device=device)

    with torch.no_grad():
        true_similarity_value = compute_rsa_score(ai_rdm_tensor, bio_rdm_tensor, device)

        for i in range(n_permutations):
            shuffled_idx = torch.randperm(num_conditions, device=device)
            shuffled_bio_tensor = bio_rdm_tensor[shuffled_idx][:, shuffled_idx]
            correlation = compute_rsa_score(ai_rdm_tensor, shuffled_bio_tensor, device)
            null_distribution_tensor[i] = correlation

    return true_similarity_value, null_distribution_tensor


def calc_p_value(true_similarity_value, null_distribution_tensor):
    n_permutations = null_distribution_tensor.shape[0]
    count_extreme = torch.sum(null_distribution_tensor >= true_similarity_value).item()
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return p_value


def process_combination(monkey, roi, noise_level, args, device):
    try:
        ai_matrix, bio_matrix = load_representations(monkey, roi, noise_level, args)
    except (ValueError, FileNotFoundError):
        return None

    ai_matrix_tensor = torch.tensor(ai_matrix, device=device, dtype=torch.float32)
    bio_matrix_tensor = torch.tensor(bio_matrix, device=device, dtype=torch.float32)
    ai_rdm_tensor, bio_rdm_tensor = generate_rdms(ai_matrix_tensor, bio_matrix_tensor)

    true_similarity_value, null_distribution_tensor = generate_null_distribution(
        ai_rdm_tensor,
        bio_rdm_tensor,
        args.n_permutations,
        device,
        args.random_seed,
    )

    p_value = calc_p_value(true_similarity_value, null_distribution_tensor)

    results_dict = {
        "monkey": monkey,
        "roi": roi,
        "noise_level": noise_level,
        "total_images": ai_matrix.shape[0],
        "rdm_metric": args.rdm_metric,
        "n_permutations": args.n_permutations,
        "random_seed": args.random_seed,
        "true_alignment_score": true_similarity_value.item(),
        "p_value": p_value,
    }

    if args.save_null_dists:
        args.null_dists_dir.mkdir(parents=True, exist_ok=True)
        noise_rounded = round(noise_level, 2)
        dist_filename = f"null_dist_{monkey}_{roi}_noise_{noise_rounded:.2f}.npy"
        dist_path = args.null_dists_dir / dist_filename
        np.save(dist_path, null_distribution_tensor.cpu().numpy())

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

    combinations = list(itertools.product(args.monkeys, args.rois, args.noise_levels))
    progress_bar = tqdm(combinations, desc="Initializing...", unit="comb")

    all_results = []

    for monkey, roi, noise_level in progress_bar:
        progress_bar.set_description(f"[{monkey} | {roi} | Noise: {noise_level}]")
        result_dict = process_combination(monkey, roi, noise_level, args, device)

        if result_dict:
            all_results.append(result_dict)
            progress_bar.set_postfix(
                score=f"{result_dict['true_alignment_score']:.4f}",
                p=f"{result_dict['p_value']:.4f}",
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
