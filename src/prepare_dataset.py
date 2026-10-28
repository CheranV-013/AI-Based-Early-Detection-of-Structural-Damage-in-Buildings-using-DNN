"""Split labeled source images into train/validation/test folders.

Expected source:
dataset/source/crack/*
dataset/source/no_crack/*

Use only images with trustworthy labels. This script does not infer labels.
"""
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "source"
DEST = ROOT / "dataset"
CLASSES = ["crack", "no_crack"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}

def main():
    random.seed(SEED)
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing {SOURCE}. Create source/crack and source/no_crack first."
        )

    gathered = {}
    for class_name in CLASSES:
        folder = SOURCE / class_name
        if not folder.exists():
            raise SystemExit(f"Missing class folder: {folder}")
        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
        if len(images) < 10:
            raise SystemExit(
                f"Only {len(images)} images found in {folder}; "
                "add labeled images before splitting."
            )
        random.shuffle(images)
        gathered[class_name] = images

    # Refuse to silently mix old split files with a new split.
    for split in SPLITS:
        path = DEST / split
        if path.exists():
            shutil.rmtree(path)

    for class_name, images in gathered.items():
        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["validation"])
        groups = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split, paths in groups.items():
            target = DEST / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(paths):
                # Prefix with class and index to avoid duplicate basenames.
                shutil.copy2(src, target / f"{class_name}_{i:06d}{src.suffix.lower()}")

    print("Dataset split complete.")
    for split in SPLITS:
        for class_name in CLASSES:
            count = len(list((DEST / split / class_name).glob("*")))
            print(f"{split:10s} {class_name:10s}: {count}")

if __name__ == "__main__":
    main()
