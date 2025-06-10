import os
from pathlib import Path

import torch
from torchvision import transforms
from moviepy.editor import ImageSequenceClip, AudioFileClip
from PIL import Image

from .train_model import Autoencoder


def load_model(model_path: str, device: str = "cpu") -> Autoencoder:
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_frames(model: Autoencoder, num_frames: int, device: str = "cpu"):
    frames = []
    noise = torch.randn(num_frames, 3, 128, 128, device=device)
    with torch.no_grad():
        outputs = model(noise).cpu()
    to_pil = transforms.ToPILImage()
    for i in range(num_frames):
        img = to_pil(outputs[i])
        frames.append(img)
    return frames


def create_video(frames, audio_path: str, output_file: str, fps: int = 24):
    frame_paths = []
    temp_dir = Path("_temp_frames")
    temp_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = temp_dir / f"frame_{i:04d}.png"
        frame.save(frame_path)
        frame_paths.append(str(frame_path))

    clip = ImageSequenceClip(frame_paths, fps=fps)
    if audio_path:
        audio = AudioFileClip(audio_path)
        clip = clip.set_audio(audio)
    clip.write_videofile(output_file)

    for p in frame_paths:
        os.remove(p)
    temp_dir.rmdir()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a music video using a trained model")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate for the video")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="output.mp4", help="Output video file")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    frames = generate_frames(model, args.frames, args.device)
    create_video(frames, args.audio, args.out, args.fps)
    print(f"Video saved to {args.out}")


if __name__ == "__main__":
    main()
