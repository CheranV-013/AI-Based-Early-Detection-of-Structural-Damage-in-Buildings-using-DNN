# AI-Based Early Detection of Structural Damage in Buildings using DNN

A college-project starter application for **image-based concrete crack screening**. It uses TensorFlow/Keras with MobileNetV2 as a frozen feature extractor and a dense neural-network (DNN) classification head, a FastAPI prediction API, SQLite inspection history, and a Streamlit dashboard.

## Important scope and safety

The recommended starter dataset (including SDNET2018) supports **cracked vs. non-cracked image classification**, not reliable minor/moderate/severe structural severity labels. Therefore, this implementation deliberately reports only:
- `crack_detected`
- `no_crack_detected`

A predicted class is not an engineering diagnosis. The system does not determine load capacity, hidden damage, building safety, or whether a building should be occupied. Any concern about structural damage should be assessed by a qualified structural engineer. Do not use this student prototype for life-safety decisions.

## Architecture

1. Upload a concrete surface image.
2. Resize to 224 × 224 RGB.
3. MobileNetV2 (ImageNet weights, frozen initially) extracts visual features.
4. Dense layers classify crack / no crack.
5. FastAPI returns class, model probability, and an explicit non-safety disclaimer.
6. Streamlit displays results and saves an inspection record to SQLite.

## Dataset

### Recommended: SDNET2018
Official repository: https://digitalcommons.usu.edu/all_datasets/48/
Paper / dataset description: https://doi.org/10.1016/j.dib.2018.11.015

SDNET2018 contains over 56,000 images of cracked and non-cracked concrete surfaces from bridge decks, walls, and pavements. Download the ZIP from the official Utah State University repository and read its README/license. It is large (about 516 MB in the repository listing).

Other option: use a concrete crack image dataset that explicitly contains `Positive` (cracked) and `Negative` (non-cracked) folders. Check the source's licence and citation requirements before reuse.

### Prepare data

The included helper expects the source data organized like this:

```text
dataset/source/
├── crack/
│   ├── image001.jpg
│   └── ...
└── no_crack/
    ├── image001.jpg
    └── ...
```

For a dataset with `Positive` and `Negative` folders, copy/rename them to `crack` and `no_crack`. For SDNET2018, use its README/annotations to identify labels; do not guess labels from image appearance or arbitrarily assign labels. Copy labeled images into the matching source folders.

Then run:

```bash
python src/prepare_dataset.py
```

This creates train/validation/test folders with a 70/15/15 split. The helper uses a fixed seed for reproducibility. For research-grade evaluation, group images by original specimen/building/source before splitting, where source metadata permits, to avoid near-duplicate leakage.

## Mac + VS Code setup

Use a recent Python version supported by your TensorFlow release. On macOS, Python 3.11 is a practical starting point; check TensorFlow's official installation page if your machine differs.

```bash
cd structural-damage-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails on Apple Silicon, follow the current TensorFlow macOS installation instructions rather than installing random incompatible versions.

## Run

### 1. Prepare data
Place images in `dataset/source/crack` and `dataset/source/no_crack`, then:

```bash
python src/prepare_dataset.py
```

### 2. Train and evaluate
```bash
python src/train.py
```

Outputs:
- `models/structural_damage_dnn.keras`
- `outputs/evaluation_metrics.json`
- `outputs/training_history.json`
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`

Training will not work until the dataset folders have images. The code prints measured results; no accuracy is promised in advance.

### 3. Start the API
In terminal 1:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/api/health

### 4. Start the dashboard
In terminal 2, with the same virtual environment active:

```bash
streamlit run app.py
```

Dashboard: http://localhost:8501

## Evaluation

The training script reports:
- held-out test accuracy
- macro F1
- precision, recall, and F1 for each class
- confusion matrix

For this screening task, inspect **crack-class recall** and false negatives carefully. Overall accuracy alone is insufficient. Keep test images out of training and validation. If multiple crops come from the same original photograph or physical specimen, keep those groups in one split.

## Database

SQLite file: `outputs/inspections.db`. The app stores timestamp, filename, predicted label, probability, and a disclaimer. Images themselves are not stored in the database.

## Suggested future work

- Compare DNN head against a custom CNN baseline using identical data splits.
- Fine-tune a small number of MobileNetV2 layers after the first training stage.
- Add Grad-CAM visualization and have a domain expert review whether highlighted regions are meaningful.
- Add image-quality checks, user authentication, audit logs, and deployment hardening.
- Only add severity labels if you obtain expert-labelled data and define a defensible engineering rubric.

## References

- Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). *SDNET2018: A concrete crack image dataset for machine learning applications*. Data in Brief, 21, 1664–1668. https://doi.org/10.1016/j.dib.2018.11.015
- TensorFlow transfer learning guide: https://www.tensorflow.org/tutorials/images/transfer_learning
