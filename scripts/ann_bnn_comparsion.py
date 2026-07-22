import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rsatoolbox
from tqdm import tqdm

BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed"
RESULTS_DATA_DIR = BASE_DATA_DIR / "results"
CSV_METADATA_PATH = PROCESSED_DATA_DIR / "things_metadata_subset.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare Artificial and Biological Representations using RSA"
    )

    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--roi", type=str, default="IT")
    parser.add_argument("--noise_degree", type=float, default=0.4)
    parser.add_argument("--rdm_metric", type=str, default="correlation")
    parser.add_argument("--compare_method", type=str, default="rho-a")
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    default_output = str(RESULTS_DATA_DIR / "rsa_permutation_results.csv")
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


def generate_plot(null_distribution, true_similarity_value, args):
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

    plt.title(f"RSA Permutation Test: {args.monkey} ({args.roi}) vs SD Noise {args.noise_degree}")
    plt.xlabel(f"Similarity Score ({args.compare_method})")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plot_filename = f"histogram_{args.monkey}_{args.roi}_noise{args.noise_degree}.png"
    plot_path = RESULTS_DATA_DIR / plot_filename
    plt.savefig(plot_path)
    plt.close()
    print(f"Successfully saved histogram to: {plot_path}")


def main():
    args = parse_arguments()

    if not CSV_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata sheet at: {CSV_METADATA_PATH}")

    print(f"Target: {args.monkey} | ROI: {args.roi} | SD Noise Step: {args.noise_degree}")

    df = pd.read_csv(CSV_METADATA_PATH)
    subset_df = df[(df["monkey"] == args.monkey) & (df["ROI"] == args.roi)]
    if subset_df.empty:
        raise ValueError(f"No metadata found for monkey '{args.monkey}' and ROI '{args.roi}'.")

    assert subset_df["response_file_name"].nunique() == 1, (
        "Multiple response files found in subset!"
    )

    image_ids = subset_df["image_id"].values.astype(int)
    response_indices = subset_df["response_file_index"].values.astype(int)
    response_file_name = subset_df.iloc[0]["response_file_name"]

    ai_file_path = PROCESSED_DATA_DIR / f"sd_mid_block_{args.noise_degree:.2f}.npy"
    ai_features = np.load(ai_file_path, mmap_mode="r")
    ai_matrix = ai_features[image_ids]

    bio_file_path = PROCESSED_DATA_DIR / response_file_name
    bio_features = np.load(bio_file_path)["representations"]
    bio_matrix = bio_features[response_indices]

    ai_dataset = rsatoolbox.data.Dataset(ai_matrix)
    bio_dataset = rsatoolbox.data.Dataset(bio_matrix)

    print("Computing RMDs (this could take a while)...")
    ai_rdm = rsatoolbox.rdm.calc_rdm(ai_dataset, method=args.rdm_metric)
    bio_rdm = rsatoolbox.rdm.calc_rdm(bio_dataset, method=args.rdm_metric)

    true_score_obj = rsatoolbox.rdm.compare(ai_rdm, bio_rdm, method=args.compare_method)
    true_similarity_value = float(true_score_obj[0][0])

    print(f"Permutation Testing ({args.n_permutations} iterations)")
    np.random.seed(args.random_seed)
    null_distribution = np.zeros(args.n_permutations)

    bio_rdm_matrix = bio_rdm.get_matrices()[0]
    num_conditions = bio_rdm_matrix.shape[0]

    for i in tqdm(range(args.n_permutations), desc="Generating Null Distribution"):
        shuffled_idx = np.random.permutation(num_conditions)
        shuffled_mat = bio_rdm_matrix[shuffled_idx, :][:, shuffled_idx]
        shuffled_bio_rdm = rsatoolbox.rdm.RDMs(np.array([shuffled_mat]))
        null_score_obj = rsatoolbox.rdm.compare(
            ai_rdm, shuffled_bio_rdm, method=args.compare_method
        )
        null_distribution[i] = float(null_score_obj[0][0])

    p_value = (np.sum(null_distribution >= true_similarity_value) + 1) / (args.n_permutations + 1)

    print("\n==============================================")
    print("                RSA RESULTS                   ")
    print("==============================================")
    print(f"Model:          Stable Diffusion (Noise: {args.noise_degree})")
    print(f"Subject:        {args.monkey} ({args.roi})")
    print(f"Total Rows:     {len(image_ids)}")
    print(f"Score Metric:   {args.compare_method}")
    print(f"True Alignment: {true_similarity_value:.4f}")
    print(f"p-value:        {p_value:.4f}")
    print("==============================================\n")

    output_csv_path = Path(args.output_csv)

    results_dict = {
        "monkey": args.monkey,
        "roi": args.roi,
        "noise_degree": args.noise_degree,
        "total_images": len(image_ids),
        "rdm_metric": args.rdm_metric,
        "compare_method": args.compare_method,
        "n_permutations": args.n_permutations,
        "random_seed": args.random_seed,
        "true_alignment_score": true_similarity_value,
        "p_value": p_value,
    }

    results_df = pd.DataFrame([results_dict])
    write_header = not output_csv_path.exists()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, mode="a", index=False, header=write_header)
    print(f"Appended results to: {output_csv_path}")

    if args.save_plot:
        generate_plot(null_distribution, true_similarity_value, args)


if __name__ == "__main__":
    main()
