import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

MASK_DATASET_PATH = Path("data/raw/face-mask-12k-images-dataset")
CELEBA_SPOOF_PATH = Path("data/raw/celeba-spoof-for-face-antispoofing")
ANTI_SPOOFING_PATH = Path("data/raw/anti-spoofing")

PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
CLASSES = ["real", "spoof", "masked"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_dirs():
    # Stage folders + final split folders are both required.
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for class_name in CLASSES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (PROCESSED_DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)


def copy_images(source_dir: Path, destination_class: str):
    if not source_dir.exists():
        print(f"Warning: {source_dir} not found. Skipping.")
        return
    destination = PROCESSED_DATA_DIR / destination_class
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    for image_file in tqdm(files, desc=f"Collecting {destination_class}"):
        shutil.copy2(image_file, destination / image_file.name)


def process_datasets():
    copy_images(MASK_DATASET_PATH / "Train" / "WithMask", "masked")
    copy_images(MASK_DATASET_PATH / "Train" / "WithoutMask", "real")
    copy_images(CELEBA_SPOOF_PATH / "live", "real")
    copy_images(CELEBA_SPOOF_PATH / "spoof", "spoof")
    copy_images(ANTI_SPOOFING_PATH / "real", "real")
    copy_images(ANTI_SPOOFING_PATH / "spoof", "spoof")


def split_data(seed=42):
    random.seed(seed)
    for class_name in CLASSES:
        source = PROCESSED_DATA_DIR / class_name
        images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        splits = {
            TRAIN_DIR / class_name: images[:train_count],
            VAL_DIR / class_name: images[train_count:train_count + val_count],
            TEST_DIR / class_name: images[train_count + val_count:],
        }

        for destination, files in splits.items():
            destination.mkdir(parents=True, exist_ok=True)
            for image_file in tqdm(files, desc=f"Splitting {class_name}"):
                target = destination / image_file.name
                if target.exists():
                    # Avoid collisions when multiple source datasets share filenames.
                    target = destination / f"{image_file.stem}_{abs(hash(str(image_file))) % 10**8}{image_file.suffix}"
                shutil.move(str(image_file), str(target))

        if source.exists() and not any(source.iterdir()):
            source.rmdir()


def main():
    create_dirs()
    process_datasets()
    split_data()
    print("\nData preprocessing complete.")
    print(f"Train: {TRAIN_DIR}")
    print(f"Validation: {VAL_DIR}")
    print(f"Test: {TEST_DIR}")


if __name__ == "__main__":
    main()
