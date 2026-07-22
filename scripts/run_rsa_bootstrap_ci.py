#!/usr/bin/env python
"""Bootstrap confidence intervals for THINGS model-brain RSA.

The reusable entry point is `compute_area_confidence_interval(area, ...)`.
The CLI writes one CSV row per area/noise level with the observed RSA score and
bootstrap confidence interval.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import rsatoolbox

from diffusion_brain_alignment.data.things_monkey_ephys import (
    _metadata,
    load_brain_response,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "diffusion_features"
if not DEFAULT_FEATURES.exists():
    DEFAULT_FEATURES = PROJECT_ROOT / "data" / "diffusion features"
DEFAULT_OUT = PROJECT_ROOT / "diffusion_brain_alignment" / "data" / "results" / "rsa_bootstrap_ci.csv"
DEFAULT_N_IMAGES = 1854


def _feature_keys(noise):
    return [f"noise_{noise:g}", f"noise_{noise:.2f}"]


def _noise_from_feature_name(path):
    return round(float(path.stem.rsplit("_", 1)[-1]), 2)


def _load_feature_npz(path, noise_levels=None):
    features = np.load(path, mmap_mode="r")
    available = [float(x) for x in features["noise_levels"]]
    if noise_levels is None:
        noise_levels = available

    out = {}
    for noise in noise_levels:
        keys = _feature_keys(noise)
        key = next((candidate for candidate in keys if candidate in features), None)
        if key is None:
            raise KeyError(f"noise={noise:g} not found in {path}. Available noise levels: {available}")
        out[float(noise)] = np.asarray(features[key], dtype=np.float32)
    return out


def _load_feature_dir(path, noise_levels=None):
    files = sorted(Path(path).glob("sd_mid_block_*.npy"))
    if not files:
        raise FileNotFoundError(f"No sd_mid_block_*.npy files found in {path}")

    available = {_noise_from_feature_name(file): file for file in files}
    if noise_levels is None:
        noise_levels = sorted(available)

    out = {}
    for noise in noise_levels:
        key = round(float(noise), 2)
        if key not in available:
            raise KeyError(f"noise={noise:g} not found in {path}. Available noise levels: {sorted(available)}")
        out[key] = np.asarray(np.load(available[key], mmap_mode="r"), dtype=np.float32)
    return out


def load_features(path, noise_levels=None):
    """Load diffusion features from a full-train directory or an older .npz bundle."""
    path = Path(path)
    if path.is_dir():
        return _load_feature_dir(path, noise_levels)
    if path.suffix == ".npz":
        return _load_feature_npz(path, noise_levels)
    if path.suffix == ".npy":
        if noise_levels is None:
            noise_levels = [_noise_from_feature_name(path)]
        if len(noise_levels) != 1:
            raise ValueError("A single .npy feature file can only be used with one noise level")
        return {round(float(noise_levels[0]), 2): np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)}
    raise ValueError(f"Unsupported feature path: {path}")


def category_from_image_name(image_name):
    return str(image_name).replace("\\", "/").split("/", 1)[0]


def first_image_per_category_trials(monkey="monkeyF", split="train", max_categories=None):
    """Return trials for the first image encountered in each THINGS category."""
    image_names = _metadata()[split]
    seen = set()
    rows = []
    names = []

    for trial, image_name in enumerate(image_names):
        category = category_from_image_name(image_name)
        if category in seen:
            continue
        seen.add(category)
        rows.append(trial)
        names.append(image_name)
        if max_categories is not None and len(rows) >= max_categories:
            break

    return {
        "monkey": monkey,
        "split": split,
        "trials": rows,
        "image_names": names,
    }


def rsa_score(brain_data, model_data, rdm_metric="correlation", compare_method="rho-a"):
    brain_rdm = rsatoolbox.rdm.calc_rdm(
        rsatoolbox.data.Dataset(np.asarray(brain_data)), method=rdm_metric
    )
    model_rdm = rsatoolbox.rdm.calc_rdm(
        rsatoolbox.data.Dataset(np.asarray(model_data)), method=rdm_metric
    )
    return float(rsatoolbox.rdm.compare(brain_rdm, model_rdm, method=compare_method).squeeze())


def bootstrap_rsa_ci(
    brain_data,
    model_data,
    n_bootstrap=200,
    ci=95,
    random_seed=42,
    rdm_metric="correlation",
    compare_method="rho-a",
    progress_every=25,
    progress_label="bootstrap",
):
    """Return observed RSA plus bootstrap CI over images.

    Rows/images are resampled with replacement. Brain and model arrays must have
    matching first dimensions.
    """
    brain_data = np.asarray(brain_data)
    model_data = np.asarray(model_data)
    if brain_data.shape[0] != model_data.shape[0]:
        raise ValueError(
            f"brain/model image counts differ: {brain_data.shape[0]} vs {model_data.shape[0]}"
        )

    rng = np.random.default_rng(random_seed)
    n_images = brain_data.shape[0]
    observed = rsa_score(brain_data, model_data, rdm_metric, compare_method)
    print(f"{progress_label}: observed score={observed:.4f}", flush=True)

    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_images, size=n_images)
        boot[i] = rsa_score(brain_data[idx], model_data[idx], rdm_metric, compare_method)
        done = i + 1
        if progress_every and (done == 1 or done % progress_every == 0 or done == n_bootstrap):
            print(f"{progress_label}: bootstrap {done}/{n_bootstrap}", flush=True)

    alpha = (100 - ci) / 2
    return {
        "score": observed,
        "ci_low": float(np.percentile(boot, alpha)),
        "ci_high": float(np.percentile(boot, 100 - alpha)),
        "boot_mean": float(np.mean(boot)),
        "boot_std": float(np.std(boot, ddof=1)),
        "n_images": int(n_images),
        "n_bootstrap": int(n_bootstrap),
        "ci": float(ci),
    }


def compute_area_confidence_interval(
    area,
    features_path=DEFAULT_FEATURES,
    monkey="monkeyF",
    split="train",
    n_images=DEFAULT_N_IMAGES,
    noise_levels=None,
    n_bootstrap=200,
    ci=95,
    random_seed=42,
    rdm_metric="correlation",
    compare_method="rho-a",
    progress_every=25,
):
    """Compute bootstrap RSA CIs for one brain area across category-first images."""
    features = load_features(features_path, noise_levels)
    n_feature_images = min(feat.shape[0] for feat in features.values())
    if n_images > n_feature_images:
        raise ValueError(f"requested {n_images} images but features contain {n_feature_images}")

    trials = first_image_per_category_trials(monkey=monkey, split=split, max_categories=n_images)
    if max(trials["trials"]) >= n_feature_images:
        raise ValueError(
            "Feature file has fewer rows than the selected category-first trial indices. "
            f"Need at least {max(trials['trials']) + 1}, found {n_feature_images}."
        )
    brain = load_brain_response(trials, roi=area)
    row_idx = np.asarray(trials["trials"], dtype=int)
    print(
        f"{area}: selected {len(row_idx)} first-image-per-category trials "
        f"(max row index {row_idx.max()})",
        flush=True,
    )

    rows = []
    for noise, model in features.items():
        label = f"{area} noise={noise:g}"
        print(f"{label}: starting", flush=True)
        result = bootstrap_rsa_ci(
            brain,
            model[row_idx],
            n_bootstrap=n_bootstrap,
            ci=ci,
            random_seed=random_seed,
            rdm_metric=rdm_metric,
            compare_method=compare_method,
            progress_every=progress_every,
            progress_label=label,
        )
        rows.append(
            {
                "monkey": monkey,
                "roi": area,
                "noise_degree": noise,
                "image_selection": "first_image_per_category",
                "rdm_metric": rdm_metric,
                "compare_method": compare_method,
                **result,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", nargs="+", required=True, help="brain area(s), e.g. IT V4 V1")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--monkey", default="monkeyF")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
    parser.add_argument("--noise", nargs="+", type=float, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--ci", type=float, default=95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--rdm-metric", default="correlation", choices=["correlation", "euclidean"])
    parser.add_argument("--compare-method", default="rho-a", choices=["rho-a", "corr_cov"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for area in args.area:
        print(f"computing bootstrap CI for {area}")
        df = compute_area_confidence_interval(
            area=area,
            features_path=args.features_path,
            monkey=args.monkey,
            split=args.split,
            n_images=args.n_images,
            noise_levels=args.noise,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
            random_seed=args.random_seed,
            rdm_metric=args.rdm_metric,
            compare_method=args.compare_method,
            progress_every=args.progress_every,
        )
        frames.append(df)
        for _, row in df.iterrows():
            print(
                f"{area} noise={row['noise_degree']:g} score={row['score']:.4f} "
                f"{args.ci:g}% CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
