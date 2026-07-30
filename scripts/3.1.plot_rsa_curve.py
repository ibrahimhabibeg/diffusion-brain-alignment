import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot RSA Alignment Curves (Model vs Brain)")

    parser.add_argument(
        "--monkeys",
        nargs="+",
        type=str,
        default=["monkeyF", "monkeyN"],
        help="List of monkey subjects to plot (e.g., monkeyF monkeyN).",
    )

    parser.add_argument(
        "--bootstrap_csv",
        type=Path,
        default=Path("../data/results/rsa_bootstrap_ci_results.csv"),
        help="Path to the Bootstrap Confidence Intervals and means.",
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
        help="Directory to save the generated plots.",
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
    """Extracts the average between-monkey ceiling for a specific ROI."""
    subset = ceiling_df[ceiling_df["ROI"] == roi]
    if subset.empty:
        return None
    return subset["rsa_score"].mean()


def main():
    args = parse_arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not all([args.bootstrap_csv.exists(), args.ceiling_csv.exists()]):
        raise FileNotFoundError(
            "One or more result CSVs are missing. Please run 2.x scripts first."
        )

    boot_df = pd.read_csv(args.bootstrap_csv)
    ceiling_df = pd.read_csv(args.ceiling_csv)

    ROI_COLORS = {
        "V1": "#4C72B0",
        "V4": "#DD8452",
        "IT": "#8172B2",
    }
    ROI_ORDER = ["V1", "V4", "IT"]

    available_monkeys = set(boot_df["monkey"].unique())
    requested_monkeys = [m for m in args.monkeys if m in available_monkeys]
    unique_monkeys = sorted(list(set(requested_monkeys)))

    if not unique_monkeys:
        raise ValueError(f"None of the requested monkeys {args.monkeys} were found in the data.")

    for monkey in unique_monkeys:
        fig, ax = plt.subplots(figsize=(9, 6))
        monkey_data = boot_df[boot_df["monkey"] == monkey]

        for roi in ROI_ORDER:
            roi_data = monkey_data[monkey_data["roi"] == roi].sort_values("noise_degree")
            if roi_data.empty:
                continue

            ceiling = get_noise_ceiling(ceiling_df, roi)

            noise_levels = roi_data["noise_degree"].to_numpy(dtype=float)
            boot_means = roi_data["boot_mean"].to_numpy(dtype=float)
            ci_low = roi_data["ci_low"].to_numpy(dtype=float)
            ci_high = roi_data["ci_high"].to_numpy(dtype=float)

            ceiling = get_noise_ceiling(ceiling_df, roi)

            if args.plot_mode == "normalized":
                if ceiling is None:
                    print(f"Warning: No noise ceiling found for {roi}. Skipping.")
                    continue
                y_vals = (boot_means / ceiling) * 100
                y_low = (ci_low / ceiling) * 100
                y_high = (ci_high / ceiling) * 100
            else:
                y_vals = boot_means
                y_low = ci_low
                y_high = ci_high

            legend_label = f"{roi} (Ceiling: {ceiling:.2f})" if ceiling else f"{roi}"
            ax.plot(
                noise_levels,
                y_vals,
                marker="o",
                markersize=6,
                linewidth=2,
                color=ROI_COLORS.get(roi, "#333333"),
                label=legend_label,
            )

            ax.fill_between(
                noise_levels,
                y_low,
                y_high,
                color=ROI_COLORS.get(roi, "#333333"),
                alpha=0.15,
                linewidth=0,
            )

        subject_name = monkey.replace("monkey", "")
        ax.set_title(f"Representational Alignment: Stable Diffusion vs. Macaque {subject_name}", fontsize=16)
        ax.set_xlabel("Stable Diffusion Noise Level", fontsize=14)
        
        if args.plot_mode == "normalized":
            ax.set_ylabel("RSA Alignment (% of Noise Ceiling)", fontsize=14)
        else:
            ax.set_ylabel("RSA Score (Spearman rho-a)", fontsize=14)
        
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(linestyle="--", alpha=0.7)

        ci_level = int(monkey_data["ci"].iloc[0]) if "ci" in monkey_data.columns else 95

        left_handles, left_labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            left_handles, 
            left_labels, 
            loc="upper right", 
            fontsize=11,
            title=f"Shaded bands denote {ci_level}% Bootstrap CI"
        )
        legend.get_title().set_fontsize(11)

        fig.tight_layout()
        
        suffix = "_raw" if args.plot_mode == "raw" else "_normalized"
        out_path = args.out_dir / f"alignment_{monkey}{suffix}.png"
        
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
