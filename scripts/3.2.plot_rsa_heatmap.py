import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot RSA Alignment Heatmap (Full Data)")

    parser.add_argument(
        "--permutation_csv",
        type=Path,
        default=Path("../data/results/rsa_permutation_results.csv"),
        help="Path to the True RSA scores computed on the full dataset.",
    )
    parser.add_argument(
        "--ceiling_csv",
        type=Path,
        default=Path("../data/results/monkey_rsa_comparison.csv"),
        help="Path to the between-monkey noise ceiling scores.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("../data/figures"),
        help="Directory to save the generated heatmaps.",
    )
    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects to plot (e.g., monkeyF monkeyN).",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        choices=["normalized", "raw"],
        default="raw",
        help="Choose 'normalized' (percentage of noise ceiling) or 'raw' (absolute RSA scores).",
    )

    return parser.parse_args()


def get_noise_ceiling(ceiling_df, roi):
    subset = ceiling_df[ceiling_df["ROI"] == roi]
    if subset.empty:
        return None
    return subset["rsa_score"].mean()


def main():
    args = parse_arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not all([args.permutation_csv.exists(), args.ceiling_csv.exists()]):
        raise FileNotFoundError("Result CSVs are missing. Please run the 2.x pipeline first.")

    scores_df = pd.read_csv(args.permutation_csv)
    ceiling_df = pd.read_csv(args.ceiling_csv)

    available_monkeys = set(scores_df["monkey"].unique())
    requested_monkeys = [m for m in args.monkeys if m in available_monkeys]
    unique_monkeys = sorted(list(set(requested_monkeys)))

    if not unique_monkeys:
        raise ValueError(f"None of the requested monkeys {args.monkeys} were found in the data.")

    ROI_ORDER = ["V1", "V4", "IT"]

    heatmaps = {}
    global_min = float("inf")
    global_max = float("-inf")

    for monkey in unique_monkeys:
        monkey_data = scores_df[scores_df["monkey"] == monkey]

        heat = monkey_data.pivot_table(
            index="roi", columns="noise_level", values="true_alignment_score", aggfunc="mean"
        )

        heat = heat.reindex(index=[r for r in ROI_ORDER if r in heat.index])
        noise_levels = sorted(heat.columns)
        heat = heat[noise_levels]

        if args.plot_mode == "normalized":
            for roi in heat.index:
                ceiling = get_noise_ceiling(ceiling_df, roi)
                if ceiling:
                    heat.loc[roi] = (heat.loc[roi] / ceiling) * 100

        heatmaps[monkey] = heat

        current_min = heat.min().min()
        current_max = heat.max().max()
        if pd.notna(current_min) and current_min < global_min:
            global_min = current_min
        if pd.notna(current_max) and current_max > global_max:
            global_max = current_max

    if args.plot_mode == "normalized":
        fmt_string = "{:.1f}"
        cbar_label = "RSA Alignment (% of Noise Ceiling)"
    else:
        fmt_string = "{:.3f}"
        cbar_label = "RSA Score (Spearman rho-a)"

    for monkey, heat in heatmaps.items():
        noise_levels = list(heat.columns)
        fig, ax = plt.subplots(figsize=(0.8 * len(noise_levels) + 3, 4))

        masked_data = np.ma.masked_invalid(heat.to_numpy(dtype=float))

        im = ax.imshow(masked_data, aspect="auto", cmap="magma", vmin=global_min, vmax=global_max)

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n:g}" for n in noise_levels])
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)

        ax.set_xlabel("Stable Diffusion Noise Level", fontsize=12)
        ax.set_ylabel("Brain Region (ROI)", fontsize=12)

        subject_name = monkey.replace("monkey", "")
        ax.set_title(
            f"Representational Alignment Heatmap: Macaque {subject_name}", fontsize=14, pad=15
        )

        fig.colorbar(im, ax=ax, label=cbar_label)

        for i, area in enumerate(heat.index):
            for j, noise in enumerate(noise_levels):
                value = heat.loc[area, noise]
                if pd.isna(value):
                    continue

                if global_max != global_min:
                    norm_val = (value - global_min) / (global_max - global_min)
                else:
                    norm_val = 0.5

                text_color = "black" if norm_val > 0.7 else "white"

                ax.text(
                    j,
                    i,
                    fmt_string.format(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        fig.tight_layout()

        suffix = "_raw" if args.plot_mode == "raw" else "_normalized"
        out_path = args.out_dir / f"heatmap_{monkey}{suffix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    main()
