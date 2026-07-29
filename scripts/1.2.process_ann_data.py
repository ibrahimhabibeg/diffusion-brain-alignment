import argparse
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
        img_path = self.images_dir / "object_images" / row["image_path"]
        image_id = int(row["image_id"])

        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)

        return tensor, image_id


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract Stable Diffusion Mid-Block Features")

    parser.add_argument(
        "--noise_levels",
        nargs="+",
        type=float,
        required=True,
        help="List of normalized noise degrees [0-1] to process (e.g., --noise_levels 0.25 0.5 0.75)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic noise generation"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images per batch")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("../data/raw/images"),
        help="Path to the images directory",
    )
    parser.add_argument(
        "--csv_metadata_path",
        type=Path,
        default=Path("../data/processed/things_metadata.csv"),
        help="Path to the metadata CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../data/processed/activations"),
        help="Path to save processed ANN outputs",
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


def get_pending_noise_levels(noise_steps, output_dir):
    """Check which noise steps already have saved UNet artifacts and skip them."""
    pending = []
    for noise in noise_steps:
        noise_rounded = round(noise, 2)
        out_path = output_dir / f"sd15_mid_block_noise_{noise_rounded:.2f}.npy"
        if not out_path.exists():
            pending.append(noise)
        else:
            print(f"Skipping noise {noise_rounded:.2f}: UNet features already exist at {out_path}")
    return pending


def compute_or_load_vae_latents(dataloader, pipe, output_dir, device, generator):
    """Bypasses or computes the 691 MB VAE latent dataset once."""
    vae_latents_path = output_dir / "vae_latents_precomputed.npy"

    if vae_latents_path.exists():
        print(f"Loading precomputed raw VAE latents from {vae_latents_path}...")
        vae_latents = np.load(vae_latents_path)
        # Convert back to PyTorch tensor format for pipeline execution
        return torch.from_numpy(vae_latents).to(device, dtype=pipe.vae.dtype)

    print("Precomputed VAE latents not found. Encoding images through VAE...")
    latent_buffer = []

    for image_tensors, _ in tqdm(dataloader, desc="VAE Encoding"):
        image_tensors = image_tensors.to(device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(image_tensors).latent_dist.sample(generator=generator)
            latents = latents * pipe.vae.config.scaling_factor
            # Offload to CPU memory immediately to safeguard GPU RAM
            latent_buffer.append(latents.cpu().numpy())

    # Combine and write the raw VAE artifact to disk
    all_vae_latents = np.concatenate(latent_buffer, axis=0)
    np.save(vae_latents_path, all_vae_latents)
    print(
        f"Saved raw precomputed VAE latents to {vae_latents_path} (Shape: {all_vae_latents.shape})"
    )

    return torch.from_numpy(all_vae_latents).to(device, dtype=pipe.vae.dtype)


def extract_unet_features_for_step(
    vae_latents, pipe, scheduler, noise_step, batch_size, output_dir, device, generator
):
    """Processes a single noise step end-to-end and checkpoints to disk immediately."""
    noise_rounded = round(noise_step, 2)
    final_output_path = output_dir / f"sd15_mid_block_noise_{noise_rounded:.2f}.npy"

    features_container = {}
    step_features_buffer = []

    def hook_fn(module, input, output):
        features_container["mid_block"] = output.detach().cpu()

    hook_handle = pipe.unet.mid_block.register_forward_hook(hook_fn)
    empty_prompt_embeds = pipe.text_encoder(
        pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
    )[0]

    timestep_val = normalize_to_raw_timestep(noise_step, scheduler)
    total_samples = vae_latents.shape[0]

    print(f"Extracting UNet mid-block features for noise step: {noise_rounded:.2f}...")

    try:
        for i in tqdm(range(0, total_samples, batch_size), desc=f"Noise {noise_rounded:.2f}"):
            latents_batch = vae_latents[i : i + batch_size]

            with torch.no_grad():
                prompt_embeds = empty_prompt_embeds.repeat(latents_batch.shape[0], 1, 1)
                timestep = torch.tensor(
                    [timestep_val] * latents_batch.shape[0], device=device, dtype=torch.long
                )

                noise = torch.randn(
                    latents_batch.shape,
                    generator=generator,
                    device=device,
                    dtype=latents_batch.dtype,
                )
                noisy_latents = pipe.scheduler.add_noise(latents_batch, noise, timestep)

                _ = pipe.unet(noisy_latents, timestep, encoder_hidden_states=prompt_embeds)

                raw_features = features_container.pop("mid_block")
                raw_features = raw_features.mean(dim=[2, 3])
                step_features_buffer.append(raw_features.numpy())

        final_array = np.concatenate(step_features_buffer, axis=0)
        np.save(final_output_path, final_array)
        print(f"Successfully saved: {final_output_path} | Shape: {final_array.shape}")

    finally:
        hook_handle.remove()


def main():
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending_noise_steps = get_pending_noise_levels(args.noise_levels, args.output_dir)
    if not pending_noise_steps:
        print("All requested noise levels have already been extracted. Process finished.")
        return
    if len(pending_noise_steps) < len(args.noise_levels):
        print(
            f"Processing only pending noise levels ({len(pending_noise_steps)} of {len(args.noise_levels)} total requested)."
        )

    if not args.csv_metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {args.csv_metadata_path}")

    master_df = pd.read_csv(args.csv_metadata_path)
    unique_img_df = (
        master_df.drop_duplicates(subset=["image_id"])
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    print(f"Total Unique Images to process: {len(unique_img_df)}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    pipe, scheduler = initialize_model(device)

    dataset = StreamingImageDataset(unique_img_df, args.images_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    vae_latents = compute_or_load_vae_latents(dataloader, pipe, args.output_dir, device, generator)

    for noise_step in pending_noise_steps:
        step_generator = torch.Generator(device=device).manual_seed(args.seed)

        extract_unet_features_for_step(
            vae_latents=vae_latents,
            pipe=pipe,
            scheduler=scheduler,
            noise_step=noise_step,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            device=device,
            generator=step_generator,
        )

    print(f"\nAll pending extractions successfully processed. Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
