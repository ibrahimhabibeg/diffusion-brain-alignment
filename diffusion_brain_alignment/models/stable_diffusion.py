from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
).to(device)

features = {}

def hook_fn(module, input, output):
    features["mid_block"] = output.detach().cpu()

hook_handle = pipe.unet.mid_block.register_forward_hook(hook_fn)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(img):
    return transform(img).to(device, dtype=pipe.vae.dtype)

class ImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return preprocess_image(self.image_list[index])

def normalize_to_raw_timestep(t_normalized, scheduler):
    t_clamped = max(0.0, min(1.0, t_normalized))
    total_timesteps = scheduler.config.num_train_timesteps
    max_index = total_timesteps - 1
    t_raw = int(t_clamped * max_index)
    return t_raw

def extract_features(images, normalized_timestep=0.5, batch_size=4):
    dataloader = DataLoader(ImageDataset(images), batch_size=batch_size, shuffle=False)
    all_features = []
    empty_prompt_embeds = pipe.text_encoder(
        pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
    )[0]
    timestep_val = normalize_to_raw_timestep(normalized_timestep, scheduler)
    try:
        for image_batch in dataloader:
            with torch.no_grad():
                latents = pipe.vae.encode(image_batch).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timestep = torch.tensor([timestep_val] * latents.shape[0], device=device, dtype=torch.long)

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
            prompt_embeds = empty_prompt_embeds.repeat(latents.shape[0], 1, 1)

            with torch.no_grad():
                _ = pipe.unet(noisy_latents, timestep, encoder_hidden_states=prompt_embeds)

            raw_features = features.pop("mid_block")  # (B, C, H, W)
            raw_features = raw_features.mean(dim=[2, 3])  # Average over spatial dimensions to get (B, C)
            all_features.append(torch.flatten(raw_features, start_dim=1))
    finally:
        hook_handle.remove()

    return torch.cat(all_features, dim=0)
