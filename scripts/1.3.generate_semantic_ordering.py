import argparse
from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Semantic Ordering from Image Folders")

    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("../data/raw/images"),
        help="Path to the downloaded image directories.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("../data/processed/semantic_ordering.csv"),
        help="Path to save the generated semantic ordering.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace SentenceTransformer model to use for embeddings.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    image_dir = args.images_dir / "object_images"
    print(f"Scanning directories in {image_dir}...")
    folder_names = [d.name for d in image_dir.iterdir() if d.is_dir()]

    if not folder_names:
        raise ValueError(f"No subdirectories found in {image_dir}. Check your data path.")

    clean_categories = [name.replace("_", " ") for name in folder_names]

    print(f"Found {len(folder_names)} unique category folders.")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading text encoder '{args.model_name}' on {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    print("Encoding category names into semantic space...")
    with torch.no_grad():
        embeddings = model.encode(clean_categories, show_progress_bar=True)

    print("Computing distance matrix and optimal leaf ordering...")
    dist_matrix = pdist(embeddings, metric="cosine")

    Z = linkage(dist_matrix, method="ward")
    Z_optimal = optimal_leaf_ordering(Z, dist_matrix)
    ordered_indices = leaves_list(Z_optimal)

    print("Saving semantic ordering to disk...")
    sort_order_mapping = {
        original_idx: sort_pos for sort_pos, original_idx in enumerate(ordered_indices)
    }

    output_df = pd.DataFrame(
        {
            "folder_name": folder_names,
            "clean_category": clean_categories,
            "semantic_sort_index": [sort_order_mapping[i] for i in range(len(folder_names))],
        }
    )

    output_df = output_df.sort_values("semantic_sort_index")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    print(f"Successfully saved ordering to: {args.output_csv}")


if __name__ == "__main__":
    main()
