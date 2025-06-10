# Creating Music Videos with Autoencoders

This project demonstrates how to build a simple pipeline for scraping sample videos, extracting frames, training an autoencoder, and generating new visuals synchronized with an audio track. The goal is to create short music videos from a trained model.

## Installation

Install Python dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

The repository provides four main scripts that should be run in the following order.

### 1. `scrape_data.py`

Downloads sample videos from a web page. Provide the URL of a page containing links or `<video>` tags. By default videos are saved under `data/raw`.

```bash
python scrape_data.py <url> --out data/raw --limit 10
```

### 2. `preprocess_data.py`

Extracts frames from each downloaded video. The default output directory is `data/frames`.

```bash
python preprocess_data.py data/raw --out data/frames --fps 1
```

### 3. `train_model.py`

Trains a simple autoencoder on the extracted frames. The frame directory must be organized for `torchvision.datasets.ImageFolder`:

```
data/frames/
    class1/
        image1.jpg
        ...
```

The script saves the trained model to `model.pth` by default.

```bash
python train_model.py data/frames --epochs 5 --batch-size 32 --out model.pth
```

### 4. `generate_output.py`

Generates a video using a trained model and an audio file.

```bash
python generate_output.py model.pth <audio_file> --frames 120 --fps 24 --out output.mp4
```

A temporary directory `_temp_frames` is created for the generated frames during this process.
