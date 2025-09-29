# banca_dati

This repository contains utilities for analysing the bankruptcy dataset composed
of Excel workbooks organised under the `tesi/Attive_complete` and
`tesi/Fallite_complete` directories.

## Installazione

Per eseguire lo script assicurati di avere Python 3.9+ e installa le
dipendenze una sola volta:

```bash
python -m pip install -r requirements.txt
```

## MLP training script

`mlp_training.py` loads every sheet from the Excel workbooks, constructs the
240-feature vectors (range `A1:X11`), and trains the requested multi-layer
perceptron:

- Input layer: 240 features
- Hidden layer 1: 300 neurons (ReLU)
- Hidden layer 2: 150 neurons (ReLU)
- Hidden layer 3: 75 neurons (ReLU)
- Hidden layer 4: 25 neurons (ReLU)
- Output layer: 3 neurons (direct costs, indirect costs, failure probability)

The script splits the dataset into training and validation subsets, trains the
model, evaluates the failure probability predictions, and saves a confusion
matrix plot together with the fitted feature and cost scalers.

### Usage

```bash
python mlp_training.py \
  --data-root /path/to/tesi \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --threshold 0.5 \
  --output-dir outputs
```

The command prints a JSON summary containing the paths of the generated
artifacts and the probability threshold used for the confusion matrix.

When running inside Google Colab, the dataset typically resides on Google
Drive.  Supply the `--mount-drive` flag to ask the script to mount Drive via
`google.colab` before loading the data:

```bash
python mlp_training.py \
  --mount-drive \
  --drive-data-path MyDrive/tesi \
  --epochs 50
```

You can adjust `--drive-mount-point` (defaults to `/content/drive`) and
`--drive-data-path` to match your Drive layout.  When `--mount-drive` is not
used, the script reads the dataset from `--data-root` as before.
