# Creating Music Videos

This repository contains a minimal pipeline for creating music videos using a simple autoencoder. The workflow is split into four scripts that handle scraping videos, extracting frames, training a model and finally generating a video.

## Prerequisites

* Python 3.8 or higher
* [`ffmpeg`](https://ffmpeg.org) must be available on your system for `moviepy` to work.

## Setting up a `uv` environment

[`uv`](https://github.com/astral-sh/uv) can be used to create an isolated Python environment and install the dependencies very quickly.

```bash
# install uv if you don't have it
pip install uv

# create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# install required packages
uv pip install -r requirements.txt
```

## Step-by-step usage

1. **Scrape videos**

   Download videos from a web page or from a list of pages:

   ```bash
   # single URL
   python scrape_data.py https://example.com/videos --out data/raw --limit 10

   # multiple URLs listed in urls.txt
   python scrape_data.py --urls-file urls.txt --out data/raw --limit 10
   ```

2. **Preprocess videos**

   Extract frames from the downloaded videos:

   ```bash
   python preprocess_data.py data/raw --out data/frames --fps 1 --subdir training
   ```

3. **Train the model**

   ```bash
   python train_model.py data/frames --epochs 5 --batch-size 32 --out models/autoencoder.pth
   ```

4. **Generate a music video**

   ```bash
   python generate_output.py models/autoencoder.pth path/to/song.mp3 --frames 120 --fps 24 --out output.mp4
   ```

## Docker usage

A simple `Dockerfile` is provided. Build the image and run the commands inside the container:

```bash
docker build -t music-video .

docker run --rm -v $(pwd):/app music-video \
    python scrape_data.py --help
```

Replace the command after the image name with any of the steps described above.

