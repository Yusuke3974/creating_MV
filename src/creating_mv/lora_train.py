import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


def build_dataloader(data_dir: str, batch_size: int = 1) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_pipeline(model_name: str, device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    pipe.to(device)
    return pipe


def apply_lora(pipe: StableDiffusionPipeline) -> StableDiffusionPipeline:
    lora_config = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out"], lora_dropout=0.1)
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.train()
    return pipe


def train_lora(
    data_dir: str,
    model_name: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    device: str = "cpu",
):
    dataloader = build_dataloader(data_dir, batch_size)
    pipe = load_pipeline(model_name, device)
    pipe = apply_lora(pipe)
    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=lr)
    scheduler = pipe.scheduler

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.no_grad():
                latents = pipe.vae.encode(imgs).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=None).sample
            loss = nn.functional.mse_loss(model_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    pipe.unet.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(output_dir)
    print(f"LoRA fine-tuned model saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a diffusion model with LoRA")
    parser.add_argument("data_dir", help="Directory with training images")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Pretrained model name or path")
    parser.add_argument("--out", default="lora_model", help="Where to save the LoRA weights")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_lora(args.data_dir, args.model, args.out, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
