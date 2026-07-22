import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rsatoolbox
import torch
from tqdm import tqdm

BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed"
RESULTS_DATA_DIR = BASE_DATA_DIR / "results"
CSV_METADATA_PATH = PROCESSED_DATA_DIR / "things_metadata_subset.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare Artificial and Biological Representations using RSA (Batch Mode)"
    )

    parser.add_argument("--monkeys", nargs="+", type=str, default=["monkeyF"], 
                        help="List of monkey subjects (e.g. monkeyF monkeyM)")
    parser.add_argument("--rois", nargs="+", type=str, default=["IT"],
                        help="List of ROIs to process (e.g. IT V1 V4)")
    parser.add_argument("--noise_degrees", nargs="+", type=float, default=[0.1],
                        help="List of noise degrees (e.g. 0.1 0.2 0.4)")
    
    parser.add_argument("--rdm_metric", type=str, default="correlation")
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    default_output = str(RESULTS_DATA_DIR / "rsa_permutation_results_gpu.csv")
    
    parser.add_argument(
        "--output_csv",
        type=str,
        default=default_output,
        help="Path to the output CSV file to append results to.",
    )

    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Optionally save a histogram of the permutation null distribution.",
    )

    return parser.parse_args()


def generate_plot(null_distribution, true_similarity_value, monkey, roi, noise_degree):
    plt.figure(figsize=(8, 6))
    plt.hist(
        null_distribution,
        bins=30,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Null Distribution",
    )

    plt.axvline(
        true_similarity_value,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"True Score: {true_similarity_value:.4f}",
    )

    plt.title(f"RSA Permutation Test: {monkey} ({roi}) vs SD Noise {noise_degree}")
    plt.xlabel("Similarity Score (rho-a)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plot_filename = f"histogram_{monkey}_{roi}_noise{noise_degree}.png"
    plot_path = RESULTS_DATA_DIR / plot_filename
    plt.savefig(plot_path)
    plt.close()


def load_representations(monkey, roi, noise_degree):
    if not CSV_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {CSV_METADATA_PATH}")

    df = pd.read_csv(CSV_METADATA_PATH)
    subset_df = df[(df["monkey"] == monkey) & (df["ROI"] == roi)]
    if subset_df.empty:
        raise ValueError(f"No metadata found")

    assert subset_df["response_file_name"].nunique() == 1, (
        "Multiple response files found in subset!"
    )

    image_ids = subset_df["image_id"].values.astype(int)
    response_indices = subset_df["response_file_index"].values.astype(int)
    response_file_name = subset_df.iloc[0]["response_file_name"]

    ai_file_path = PROCESSED_DATA_DIR / f"sd_mid_block_{noise_degree:.2f}.npy"
    ai_features = np.load(ai_file_path, mmap_mode="r")
    ai_matrix = ai_features[image_ids]

    bio_file_path = PROCESSED_DATA_DIR / response_file_name
    bio_features = np.load(bio_file_path)["representations"]
    bio_matrix = bio_features[response_indices]

    return ai_matrix, bio_matrix


def generate_rdms(ai_matrix, bio_matrix, rdm_metric):
    ai_dataset = rsatoolbox.data.Dataset(ai_matrix)
    bio_dataset = rsatoolbox.data.Dataset(bio_matrix)

    ai_rdm = rsatoolbox.rdm.calc_rdm(ai_dataset, method=rdm_metric)
    bio_rdm = rsatoolbox.rdm.calc_rdm(bio_dataset, method=rdm_metric)

    ai_rdm_matrix = ai_rdm.get_matrices()[0]
    bio_rdm_matrix = bio_rdm.get_matrices()[0]
    return ai_rdm_matrix, bio_rdm_matrix


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


def generate_null_distribution(ai_rdm_matrix, bio_rdm_matrix, n_permutations, device, random_seed):
    num_conditions = bio_rdm_matrix.shape[0]

    torch.manual_seed(random_seed)
    ai_rdm_tensor = torch.tensor(ai_rdm_matrix, dtype=torch.float32, device=device)
    bio_rdm_tensor = torch.tensor(bio_rdm_matrix, dtype=torch.float32, device=device)

    null_distribution_tensor = torch.zeros(n_permutations, device=device)

    with torch.no_grad():
        true_similarity_value = compute_rsa_score(ai_rdm_tensor, bio_rdm_tensor, device).item()

        for i in range(n_permutations):
            shuffled_idx = torch.randperm(num_conditions, device=device)
            shuffled_bio_tensor = bio_rdm_tensor[shuffled_idx][:, shuffled_idx]
            correlation = compute_rsa_score(ai_rdm_tensor, shuffled_bio_tensor, device)
            null_distribution_tensor[i] = correlation
            
    null_distribution = null_distribution_tensor.cpu().numpy()
    return true_similarity_value, null_distribution


def calc_p_value(true_similarity_value, null_distribution):
    p_value = (np.sum(null_distribution >= true_similarity_value) + 1) / (
        len(null_distribution) + 1
    )
    return p_value


def process_combination(monkey, roi, noise_degree, args, device):
    """
    Handles the end-to-end execution of loading, matrix generation, 
    permutation testing, and saving results for a single parameter combination.
    """
    try:
        ai_matrix, bio_matrix = load_representations(monkey, roi, noise_degree)
    except ValueError:
        return None
    
    ai_rdm_matrix, bio_rdm_matrix = generate_rdms(ai_matrix, bio_matrix, args.rdm_metric)

    true_similarity_value, null_distribution = generate_null_distribution(
        ai_rdm_matrix,
        bio_rdm_matrix,
        args.n_permutations,
        device,
        args.random_seed,
    )
    
    p_value = calc_p_value(true_similarity_value, null_distribution)

    output_csv_path = Path(args.output_csv)

    results_dict = {
        "monkey": monkey,
        "roi": roi,
        "noise_degree": noise_degree,
        "total_images": ai_matrix.shape[0],
        "rdm_metric": args.rdm_metric,
        "n_permutations": args.n_permutations,
        "random_seed": args.random_seed,
        "true_alignment_score": true_similarity_value,
        "p_value": p_value,
    }

    results_df = pd.DataFrame([results_dict])
    write_header = not output_csv_path.exists()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, mode="a", index=False, header=write_header)

    if args.save_plot:
        generate_plot(null_distribution, true_similarity_value, monkey, roi, noise_degree)
        
    return true_similarity_value, p_value


def main():
    args = parse_arguments()

    if not CSV_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {CSV_METADATA_PATH}")

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
    
    for monkey, roi, noise_degree in progress_bar:
        progress_bar.set_description(f"[{monkey} | {roi} | Noise: {noise_degree}]")
        result = process_combination(monkey, roi, noise_degree, args, device)        
        if result:
            true_sim, p_val = result
            progress_bar.set_postfix(score=f"{true_sim:.4f}", p=f"{p_val:.4f}")
        else:
            progress_bar.set_postfix(status="Skipped (No Data)")


if __name__ == "__main__":
    main()