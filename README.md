# creating_MV

## Project Goals
This repository demonstrates a simple pipeline for processing video files and training
an autoencoder on the extracted frames.

## Directory Overview
- `src/` - Source code and modules.
- `data/` - Input or processed datasets.
- `models/` - Saved model files or checkpoints.
- `scripts/` - Utility scripts for running experiments or training.

## Basic Setup
1. Clone the repository.
2. (Optional) Create and activate a Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.

## Preprocessing Videos
Use `preprocess_data.py` to convert videos into individual frames. Frames are
stored inside a subdirectory (by default `default`) under the given output
folder. Example:

```bash
python preprocess_data.py /path/to/videos --out data/frames --fps 1
```

The above command will create frames inside `data/frames/default/`.

## Training
After extracting frames, train the autoencoder using `train_model.py`. Provide
the path to the parent frame directory (the script expects class
subdirectories and treats `default` as the only class):

```bash
python train_model.py data/frames --epochs 5
```

The trained model will be saved to `model.pth` by default.
