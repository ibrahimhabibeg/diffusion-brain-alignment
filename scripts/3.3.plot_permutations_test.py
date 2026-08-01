import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot Permutation Test Histograms")

    parser.add_argument(
        "--permutation_csv",
        type=Path,
        default=Path("../data/results/rsa_permutation_results.csv"),
        help="Path to the True RSA scores and p-values.",
    )
    parser.add_argument(
        "--null_dist_dir",
        type=Path,
        default=Path("../data/results/null_distributions"),
        help="Directory containing the saved permutation null distributions (.npy files).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("../data/figures"),
        help="Directory to save the generated histograms.",
    )
    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects to plot (e.g., monkeyF monkeyN).",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        type=str,
        default=["V1", "V4", "IT"],
        help="List of ROIs to plot (e.g., V1 V4 IT).",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.null_dist_dir.exists() or not any(args.null_dist_dir.iterdir()):
        raise FileNotFoundError(
            f"Null distributions not found in {args.null_dist_dir}.\n"
            "Please rerun the permutation script with the save flag enabled:\n"
            "> python 2.1.run_permutation_test.py --save_null_dists"
        )

    if not args.permutation_csv.exists():
        raise FileNotFoundError(f"Results CSV missing at {args.permutation_csv}.")

    df = pd.read_csv(args.permutation_csv)

    available_monkeys = set(df["monkey"].unique())
    requested_monkeys = sorted([m for m in args.monkeys if m in available_monkeys])

    available_rois = set(df["roi"].unique())
    requested_rois = sorted([r for r in args.rois if r in available_rois])

    if not requested_monkeys or not requested_rois:
        raise ValueError("No matching monkeys or ROIs found in the dataset based on your filters.")

    for monkey in requested_monkeys:
        for roi in requested_rois:
            subset = df[(df["monkey"] == monkey) & (df["roi"] == roi)].sort_values("noise_level")  # type: ignore
            if subset.empty:
                continue

            noise_levels = subset["noise_level"].tolist()
            n_plots = len(noise_levels)

            ncols = min(3, n_plots)
            nrows = math.ceil(n_plots / ncols)

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False
            )
            axes = axes.flatten()

            subject_name = monkey.replace("monkey", "")
            fig.suptitle(
                f"Permutation Null Distributions: Macaque {subject_name} | {roi}",
                fontsize=18,
                fontweight="bold",
                y=1.02,
            )

            for idx, noise in enumerate(noise_levels):
                ax = axes[idx]
                row_data = subset[subset["noise_level"] == noise].iloc[0]

                true_score = float(row_data["true_alignment_score"])
                p_value = float(row_data["p_value"])

                npy_filename = f"null_dist_{monkey}_{roi}_noise_{noise:.2f}.npy"
                npy_path = args.null_dist_dir / npy_filename

                if not npy_path.exists():
                    ax.text(0.5, 0.5, f"Missing:\n{npy_filename}", ha="center", va="center")
                    ax.set_title(f"Noise: {noise:.2f}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                null_dist = np.load(npy_path)

                ax.hist(
                    null_dist,
                    bins=30,
                    density=False,
                )

                ax.axvline(
                    x=true_score,
                    color="red",
                    linewidth=2.5,
                    linestyle="--",
                    label=f"True Score ({true_score:.3f})",
                )

                ax.set_title(f"Stable Diffusion Noise: {noise:.2f}", fontsize=12)
                ax.set_xlabel("RSA Score (rho-a)", fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
                ax.text(
                    0.05,
                    0.90,
                    p_text,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="#333333",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )
                ax.legend(loc="upper right", fontsize=9)

            for empty_idx in range(n_plots, len(axes)):
                fig.delaxes(axes[empty_idx])

            fig.tight_layout()

            out_file = args.out_dir / f"permutations_{monkey}_{roi}.png"
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved permutation histogram: {out_file}")


if __name__ == "__main__":
    main()
