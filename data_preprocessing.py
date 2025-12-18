import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
# IMPORTANT: These paths must match the folder names created after unzipping.

# --- Paths for Mask Detection ---
MASK_DATASET_PATH = 'data/raw/face-mask-12k-images-dataset'

# --- Paths for Spoof Detection ---
CELEBA_SPOOF_PATH = 'data/raw/celeba-spoof-for-face-antispoofing'
ANTI_SPOOFING_PATH = 'data/raw/anti-spoofing'

# Output paths
PROCESSED_DATA_DIR = 'data/processed'
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classes
CLASSES = ['real', 'spoof', 'masked']

def create_dirs():
    """Create the necessary directories for processed data."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    print("Directory structure created.")

def process_mask_dataset():
    """Process the Face Mask Dataset."""
    print("Processing Face Mask Dataset...")
    base_path = MASK_DATASET_PATH
    if not os.path.exists(base_path):
        print(f"Warning: Mask dataset path not found at {base_path}. Skipping.")
        return

    # Process masked faces
    masked_path = os.path.join(base_path, 'Train', 'WithMask')
    if os.path.exists(masked_path):
        for img_file in tqdm(os.listdir(masked_path), desc="Moving masked faces"):
            shutil.copy(os.path.join(masked_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'masked'))

    # Process real faces (without mask)
    real_path = os.path.join(base_path, 'Train', 'WithoutMask')
    if os.path.exists(real_path):
        for img_file in tqdm(os.listdir(real_path), desc="Moving real (unmasked) faces"):
            shutil.copy(os.path.join(real_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'real'))

def process_celeba_spoof_dataset():
    """Process the CelebA-Spoof dataset."""
    print("Processing CelebA-Spoof Dataset...")
    base_path = CELEBA_SPOOF_PATH
    if not os.path.exists(base_path):
        print(f"Warning: CelebA-Spoof dataset path not found at {base_path}. Skipping.")
        return

    # Process real faces (live)
    live_path = os.path.join(base_path, 'live')
    if os.path.exists(live_path):
        for img_file in tqdm(os.listdir(live_path), desc="Moving CelebA real faces"):
            shutil.copy(os.path.join(live_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'real'))

    # Process spoof faces
    spoof_path = os.path.join(base_path, 'spoof')
    if os.path.exists(spoof_path):
        for img_file in tqdm(os.listdir(spoof_path), desc="Moving CelebA spoof faces"):
            shutil.copy(os.path.join(spoof_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'spoof'))

def process_anti_spoofing_dataset():
    """Process the Anti-Spoofing Dataset."""
    print("Processing Anti-Spoofing Dataset...")
    base_path = ANTI_SPOOFING_PATH
    if not os.path.exists(base_path):
        print(f"Warning: Anti-Spoofing dataset path not found at {base_path}. Skipping.")
        return

    # Process real faces
    real_path = os.path.join(base_path, 'real')
    if os.path.exists(real_path):
        for img_file in tqdm(os.listdir(real_path), desc="Moving Anti-Spoofing real faces"):
            shutil.copy(os.path.join(real_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'real'))

    # Process spoof faces
    spoof_path = os.path.join(base_path, 'spoof')
    if os.path.exists(spoof_path):
        for img_file in tqdm(os.listdir(spoof_path), desc="Moving Anti-Spoofing spoof faces"):
            shutil.copy(os.path.join(spoof_path, img_file), os.path.join(PROCESSED_DATA_DIR, 'spoof'))

def split_data():
    """Split the gathered data into train, validation, and test sets."""
    print("Splitting data into train, validation, and test sets...")
    for class_name in CLASSES:
        class_path = os.path.join(PROCESSED_DATA_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: No data found for class '{class_name}'. Skipping split.")
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        
        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        for f in tqdm(train_files, desc=f"Moving {class_name} train"):
            shutil.move(os.path.join(class_path, f), os.path.join(TRAIN_DIR, class_name, f))
        for f in tqdm(val_files, desc=f"Moving {class_name} val"):
            shutil.move(os.path.join(class_path, f), os.path.join(VAL_DIR, class_name, f))
        for f in tqdm(test_files, desc=f"Moving {class_name} test"):
            shutil.move(os.path.join(class_path, f), os.path.join(TEST_DIR, class_name, f))
        
        # Clean up the now-empty class folder
        os.rmdir(class_path)

if __name__ == '__main__':
    create_dirs()
    
    # --- Run the processing functions for all three datasets ---
    process_mask_dataset()
    process_celeba_spoof_dataset()
    process_anti_spoofing_dataset()
    
    # --- Split the collected data into final train/val/test folders ---
    split_data()
    
    print("\nData preprocessing complete!")
    print(f"Data has been split into:")
    print(f"- Train: {TRAIN_DIR}")
    print(f"- Validation: {VAL_DIR}")
    print(f"- Test: {TEST_DIR}")