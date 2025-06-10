# Creating Music Videos

This repository contains a simple pipeline for creating music videos using
scraped video data and a small generative model. The workflow consists of four
steps:

1. **Scrape training data** – download sample videos from a web page.
2. **Preprocess the data** – extract image frames from those videos.
3. **Train a model** – train a lightweight autoencoder on the frames.
4. **Generate output** – produce new frames with the model and combine them
   with an audio track to create a music video.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download videos**

   ```bash
   python scrape_data.py "https://example.com/videos" --out data/raw --limit 5
   ```

   Replace the URL with a page that contains video files you are permitted to
   download.
   Ensure that scraping this site is allowed by its terms of service.

2. **Extract frames**

   ```bash
   python preprocess_data.py data/raw --out data/frames --fps 1
   ```

3. **Train the autoencoder**

   ```bash
   python train_model.py data/frames --epochs 10 --device cpu --out model.pth
   ```

4. **Generate a music video**

   ```bash
   python generate_output.py model.pth path/to/music.mp3 --out output.mp4
   ```

The final video will be saved as `output.mp4`.
