import os
from pathlib import Path
try:
    from moviepy.editor import VideoFileClip
except ModuleNotFoundError:  # Compatibility with minimal moviepy packages
    from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_frames(video_path: str, output_dir: str, fps: int = 1, subdir: str = "default"):
    """Extract frames from a video.

    Parameters
    ----------
    video_path : str
        Path to a video file.
    output_dir : str
        Directory where frames will be saved.
    fps : int, optional
        Frames per second to extract, by default 1.
    """
    output_path = Path(output_dir) / subdir
    output_path.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    for t in range(0, duration * fps):
        frame_time = t / fps
        frame = clip.get_frame(frame_time)
        frame_img = os.path.join(output_path, f"{Path(video_path).stem}_{t:04d}.jpg")
        from PIL import Image

        Image.fromarray(frame).save(frame_img)
    clip.close()


def process_directory(input_dir: str, output_dir: str, fps: int = 1, subdir: str = "default"):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".mp4", ".mov", ".webm")):
            extract_frames(os.path.join(input_dir, fname), output_dir, fps, subdir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("input_dir", help="Directory with downloaded videos")
    parser.add_argument("--out", default="data/frames", help="Where to save frames")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    parser.add_argument("--subdir", default="default", help="Subdirectory under the output directory where frames will be stored")
    args = parser.parse_args()

    process_directory(args.input_dir, args.out, args.fps, args.subdir)


if __name__ == "__main__":
    main()
