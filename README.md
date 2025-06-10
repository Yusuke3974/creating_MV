# Creating Music Videos

This repository contains a minimal pipeline for generating a music video using a simple
autoencoder.  The implementation is split across four modules found under
`src/creating_mv` and tiny launcher scripts in the `scripts` directory.

## Directory layout

```
src/creating_mv/      Python modules implementing the pipeline
scripts/              Command line entry points
models/               Saved model checkpoints
data/                 Raw videos and extracted frames
```

## Installation

The project relies on Python 3.8 or later and `ffmpeg` for `moviepy`.
Dependencies are listed in `requirements.txt` and `pyproject.toml`.
With [uv](https://github.com/astral-sh/uv) you can install them quickly:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Regular `pip` also works if you prefer.

## Usage

1. **Download videos**

```bash
   scripts/scrape_data https://example.com/videos --out data/raw --limit 10
```

   To fetch several pages in parallel, provide a file of URLs to
   `scripts/multi_scrape`:

   ```bash
   scripts/multi_scrape URL.txt --out data/raw --limit 10 --workers 4
   ```

2. **Extract frames**

   ```bash
   scripts/preprocess_data data/raw --out data/frames --fps 1 --subdir training
   ```

3. **Train the autoencoder**

   ```bash
   scripts/train_model data/frames --epochs 5 --batch-size 32 --out models/autoencoder.pth
   ```

4. **Generate a music video**

   ```bash
   scripts/generate_output models/autoencoder.pth path/to/song.mp3 --frames 120 --fps 24 --out output.mp4
   ```

The above commands assume you are running them from the repository root. Feel free to
adjust the paths as needed.

A `Dockerfile` is also provided for running the pipeline inside a container.
