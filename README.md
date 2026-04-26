# Fashion Classifier Gradio App

This project compares multiple Fashion-MNIST classifiers in a single Gradio interface.

Current models shown in the app:
- `ANN`
- `Optimized ANN`
- `Hyperparameterized ANN`
- `CNN`
- `VGG16 Transfer Learning`

## Project Structure

- `gradio_app.py` contains the Gradio UI and inference logic.
- `ann_trained_model/ann.ipynb` contains the base ANN training notebook.
- `ann_optimised_model/optimised_ann.ipynb` contains the optimized ANN notebook.
- `hyperparameterised_ann_model/hyperparameterised_ann.ipynb` contains the hyperparameter tuning notebook.
- `cnn/cnn.ipynb` contains the CNN training notebook.
- `transfer_learning/transfer_learning.ipynb` contains the VGG16 transfer learning notebook.

## Included Model Files

These model weights are included in the GitHub repo:
- `pt_file/ann_model.pt`
- `pt_file/ann_optimised_model (1).pt`
- `pt_file/ann_hyperparameterimised_model.pt`
- `pt_file/cnn_model (6).pt`

## Local Files Required

The following files are intentionally not tracked in GitHub:

- `dataset/fmnist_small.csv`
- `fashion-mnist_train.csv`
- `pt_file/vgg16_model.pt`

They are ignored by `.gitignore` because:
- the CSV files are local dataset assets
- `vgg16_model.pt` is too large for a normal GitHub push

If you clone this repo and want the full app to run, place these files in exactly these locations:

```text
fashion_classifier/
├── dataset/
│   └── fmnist_small.csv
├── pt_file/
│   └── vgg16_model.pt
└── fashion-mnist_train.csv
```

## Setup

Create and activate a virtual environment, then install the dependencies you use for the notebooks/app.

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision pandas pillow gradio
```

## Run The App

```powershell
python gradio_app.py
```

The Gradio UI will launch locally and display predictions from all configured models side by side.

## Notes

- The app expects Fashion-MNIST samples in `dataset/fmnist_small.csv`.
- The VGG16 model uses different preprocessing from the ANN/CNN models:
  - grayscale image converted to RGB
  - resized to `256`
  - center cropped to `224`
  - normalized with ImageNet statistics
- If `pt_file/vgg16_model.pt` is missing, the transfer learning path in the app will not load.

## GitHub Notes

This repo is configured to ignore:
- CSV files
- Python cache files
- `pt_file/vgg16_model.pt`

That keeps the GitHub repo lighter, but means a fresh clone does not include every local runtime asset by default.
