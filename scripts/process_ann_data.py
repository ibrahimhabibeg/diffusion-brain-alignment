import argparse
import os
from pathlib import Path

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class StreamingImageDataset(Dataset):
    def __init__(self, unique_df, images_dir):
        self.df = unique_df
        self.images_dir = Path(images_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = self.images_dir / row["image_path"]
        image_id = int(row["image_id"])

        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)

        return tensor, image_id


def parse_arguments():
    BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    RAW_DATA_DIR = BASE_DATA_DIR / "raw"
    PROCESSED_DATA_DIR_DEFAULT = BASE_DATA_DIR / "processed"
    IMAGES_DIR_DEFAULT = RAW_DATA_DIR / "images"
    CSV_METADATA_PATH_DEFAULT = PROCESSED_DATA_DIR_DEFAULT / "things_metadata.csv"

    parser = argparse.ArgumentParser(description="Extract Stable Diffusion Mid-Block Features")

    parser.add_argument(
        "--noise_degree",
        type=float,
        required=True,
        help="Normalized noise degree float [0-1] (Mandatory)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images per batch")
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Number of batches to process before appending to the file",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=str(IMAGES_DIR_DEFAULT),
        help="Path to the images directory",
    )
    parser.add_argument(
        "--csv_metadata_path",
        type=str,
        default=str(CSV_METADATA_PATH_DEFAULT),
        help="Path to the metadata CSV file",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR_DEFAULT),
        help="Path to save processed outputs",
    )

    return parser.parse_args()


def normalize_to_raw_timestep(t_normalized, scheduler):
    t_clamped = max(0.0, min(1.0, t_normalized))
    total_timesteps = scheduler.config.num_train_timesteps
    return int(t_clamped * (total_timesteps - 1))


def initialize_model(device):
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    ).to(device)
    pipe.vae.eval()
    pipe.unet.eval()
    return pipe, scheduler


def append_to_numpy_file(filepath, buffer_list):
    if not buffer_list:
        return

    new_data = np.vstack(buffer_list)

    if filepath.exists():
        existing_data = np.load(filepath)
        combined_data = np.vstack([existing_data, new_data])
        np.save(filepath, combined_data)
    else:
        np.save(filepath, new_data)


def scan_and_filter_progress(final_output_path, csv_metadata_path):
    csv_path = Path(csv_metadata_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")

    master_df = pd.read_csv(csv_path)
    unique_img_df = (
        master_df.drop_duplicates(subset=["image_id"])
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    processed_count = 0
    if final_output_path.exists():
        existing_data = np.load(final_output_path)
        processed_count = existing_data.shape[0]
        print(f"Resuming from existing data: {processed_count} processed samples")

    remaining_df = unique_img_df.iloc[processed_count:].reset_index(drop=True)
    print(
        f"Total Unique Images: {len(unique_img_df)} | Already Processed: {processed_count} | Remaining: {len(remaining_df)}"
    )
    return unique_img_df, remaining_df


def extract_features(dataloader, pipe, scheduler, args, final_output_path, device):
    features_container = {}
    feature_buffer = []

    def hook_fn(module, input, output):
        features_container["mid_block"] = output.detach().cpu()

    hook_handle = pipe.unet.mid_block.register_forward_hook(hook_fn)

    empty_prompt_embeds = pipe.text_encoder(
        pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
    )[0]
    timestep_val = normalize_to_raw_timestep(args.noise_degree, scheduler)

    print("Starting extraction pipeline...")
    try:
        for batch_idx, (image_tensors, _) in enumerate(
            tqdm(dataloader, desc="Extracting Features")
        ):
            image_tensors = image_tensors.to(device, dtype=pipe.vae.dtype)

            with torch.no_grad():
                latents = pipe.vae.encode(image_tensors).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timestep = torch.tensor(
                    [timestep_val] * latents.shape[0], device=device, dtype=torch.long
                )
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
                prompt_embeds = empty_prompt_embeds.repeat(latents.shape[0], 1, 1)

                _ = pipe.unet(noisy_latents, timestep, encoder_hidden_states=prompt_embeds)

            raw_features = features_container.pop("mid_block")
            raw_features = raw_features.mean(dim=[2, 3])
            feature_buffer.append(raw_features.numpy())

            if (batch_idx + 1) % args.save_every == 0:
                append_to_numpy_file(final_output_path, feature_buffer)
                feature_buffer.clear()
                tqdm.write(f"Saved progress to disk at batch {batch_idx + 1}")

        if feature_buffer:
            append_to_numpy_file(final_output_path, feature_buffer)
            feature_buffer.clear()
            tqdm.write("Final buffer flushed to disk.")
    except Exception as e:
        tqdm.write(f"Error during feature extraction: {e}")
    finally:
        hook_handle.remove()


def main():
    args = parse_arguments()
    processed_data_dir = Path(args.processed_data_dir)
    os.makedirs(processed_data_dir, exist_ok=True)
    noise_rounded = round(args.noise_degree, 2)
    final_output_path = processed_data_dir / f"sd_mid_block_{noise_rounded:.2f}.npy"
    _, remaining_df = scan_and_filter_progress(final_output_path, args.csv_metadata_path)

    if len(remaining_df) > 0:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")
        pipe, scheduler = initialize_model(device)
        dataset = StreamingImageDataset(remaining_df, args.images_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device == "cuda" else False,
        )
        extract_features(dataloader, pipe, scheduler, args, final_output_path, device)
        print(f"Extraction complete. Target file is located at {final_output_path}")
    else:
        print("All image layers have already been extracted. Process finished.")


if __name__ == "__main__":
    main()
