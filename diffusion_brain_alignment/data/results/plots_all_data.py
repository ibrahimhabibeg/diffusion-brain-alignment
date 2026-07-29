"""
Builds "model-brain alignment normalized by noise ceiling" plots,
one per monkey (monkeyF, monkeyN), each showing V1, V4, and IT as
separate lines on the same axes.

Inputs:
    - Monkeys_Results.xlsx        -> between-monkey noise ceiling per ROI
    - rsa_permutation_results_gpu.csv -> model-brain RSA scores per
      monkey / ROI / noise_degree

Output:
    - alignment_monkeyF.png
    - alignment_monkeyN.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
ceiling_df = pd.read_excel("Monkeys Results.xlsx")
scores_df = pd.read_csv("rsa_permutation_results_gpu.csv")

# Between-monkey ceiling: use the rho-a / correlation row for each ROI
ceiling_sub = ceiling_df[
    (ceiling_df["compare_method"] == "rho-a")
    & (ceiling_df["rdm_metric"] == "correlation")
]
ceiling_by_roi = dict(zip(ceiling_sub["ROI"], ceiling_sub["rsa_score"]))

# ---------------------------------------------------------------
# 1a. Load bootstrap CI results (for the shaded CI band + significance stars)
# ---------------------------------------------------------------
_boot_cols = [
    "monkey", "roi", "noise_degree", "rdm_metric", "compare_method",
    "random_seed", "score", "ci_low", "ci_high", "boot_mean", "boot_std",
    "n_images", "n_bootstrap_actual", "n_bootstrap", "ci",
]
boot_df = pd.read_csv(
    "rsa_bootstrap_ci_results_gpu.csv", skiprows=2, names=_boot_cols
)
boot_df = boot_df.drop_duplicates(
    subset=["monkey", "roi", "noise_degree"], keep="last"
)

# ---------------------------------------------------------------
# 2. Plot settings
# ---------------------------------------------------------------
ROI_COLORS = {
    "V1": "#4C72B0",
    "V4": "#DD8452",
    "IT": "#8172B2",
}
ROI_ORDER = ["V1", "V4", "IT"]
MONKEYS = ["monkeyF", "monkeyN"]

for monkey in MONKEYS:
    fig, ax = plt.subplots(figsize=(9, 6))

    monkey_data = scores_df[scores_df["monkey"] == monkey]
    monkey_boot = boot_df[boot_df["monkey"] == monkey]

    for roi in ROI_ORDER:
        roi_data = monkey_data[monkey_data["roi"] == roi].sort_values("noise_degree")
        if roi_data.empty:
            continue

        ceiling = ceiling_by_roi.get(roi)
        if ceiling is None:
            continue

        true_scores = roi_data["true_alignment_score"]
        print(f"{monkey} {roi}: max true_scores = {max(true_scores.values)}, min true_scores = {min(true_scores.values)}")
        pct_of_ceiling = true_scores / ceiling * 100

        ax.plot(
            roi_data["noise_degree"],
            pct_of_ceiling,
            marker="o",
            markersize=6,
            linewidth=2,
            color=ROI_COLORS[roi],
            label=f"{roi} (Spearman rho_a = [{min(true_scores.values):.3f}, {max(true_scores.values):.3f}], ceiling = {ceiling:.3f})",
        )

        roi_boot = monkey_boot[monkey_boot["roi"] == roi].sort_values("noise_degree")
 

        if not roi_boot.empty:
            ci_low_pct = roi_boot["ci_low"] / ceiling * 100
            ci_high_pct = roi_boot["ci_high"] / ceiling * 100
            ax.fill_between(
                roi_boot["noise_degree"],
                ci_low_pct,
                ci_high_pct,
                color=ROI_COLORS[roi],
                alpha=0.2,
                linewidth=0,
            )

    ax.set_title(
        f"{monkey}: model-brain alignment normalized by noise ceiling",
        fontsize=18,
    )
    ax.set_xlabel("noise level", fontsize=14)
    ax.set_ylabel("% of between-monkey ceiling", fontsize=14)
  
    ax.set_ylim(bottom=min(-5, ax.get_ylim()[0]), top=65)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.tick_params(axis="both", labelsize=12)

    ax.grid(linestyle="--", alpha=0.7)

    left_handles, left_labels = ax.get_legend_handles_labels()

    ax.legend(left_handles, left_labels, loc="upper right", fontsize=12)

    fig.tight_layout()
    out_path = f"alignment_{monkey}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")