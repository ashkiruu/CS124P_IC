# =============================================================================
# Waste Classification - Data Preprocessing Pipeline (Modern TF API)
# Uses tf.data + tf.keras.utils.image_dataset_from_directory
# Compatible with TensorFlow 2.9+ / Keras 3+
# =============================================================================

import os
import shutil
import random
import logging
from pathlib import Path
from PIL import Image

import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (Updated with Absolute Paths)
# =============================================================================

# Calculate the absolute path to the project root (CS124P_IC)
# This assumes the script is located in CS124P_IC/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # Convert Paths to strings for TensorFlow compatibility
    "dataset_root": str(PROJECT_ROOT / "archive" / "standardized_384"), 
    "split_output": str(PROJECT_ROOT / "dataset_split"), 
    "img_size":      (224, 224),          
    "batch_size":    32,
    "splits": {
        "train": 0.70,
        "val":   0.15,
        "test":  0.15,
    },
    "seed": 42,
}


# =============================================================================
# STEP 1: VALIDATE & FILTER CORRUPTED IMAGES
# =============================================================================

def is_valid_image(filepath: str) -> bool:
    """
    Verify image integrity using Pillow before it enters the pipeline.
    Prevents TensorFlow from crashing on corrupted files at runtime.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, Image.DecompressionBombError):
        logger.warning(f"Corrupted image skipped: {filepath}")
        return False


def collect_valid_images(class_dir: str) -> list:
    """
    Return only valid image paths from a class directory.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if Path(f).suffix.lower() in valid_extensions
        and is_valid_image(os.path.join(class_dir, f))
    ]


# =============================================================================
# STEP 2: SPLIT DATASET INTO TRAIN / VAL / TEST FOLDERS
# =============================================================================

def split_dataset(config: dict) -> str:
    """
    Copies images from the raw dataset root into train/val/test folders.
    """
    src_root = config["dataset_root"]
    dst_root = config["split_output"]
    splits   = config["splits"]
    seed     = config["seed"]

    assert abs(sum(splits.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Robust check: Ensure destination exists and has actual content
    if os.path.exists(dst_root) and any(os.scandir(dst_root)):
        logger.info(f"Split directory already populated at '{dst_root}'. Skipping split phase.")
        return dst_root
    
    # Clean up empty/broken destination folder before starting
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)

    logger.info(f"Checking source dataset at: '{src_root}'")
    if not os.path.exists(src_root):
         raise FileNotFoundError(f"Source dataset not found at: {src_root}. Please ensure 'archive/standardized_384' exists.")

    random.seed(seed)
    class_names = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ])

    if not class_names:
        raise ValueError(f"No class subfolders found in '{src_root}'.")

    logger.info(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(src_root, class_name)
        images = collect_valid_images(class_path)
        random.shuffle(images)

        n_total = len(images)
        if n_total == 0:
             logger.warning(f"No valid images found for class '{class_name}' in {class_path}.")
             continue

        n_train = int(n_total * splits["train"])
        n_val   = int(n_total * splits["val"])

        subsets = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],   # absorbs rounding remainder
        }

        for subset_name, subset_images in subsets.items():
            dest_dir = os.path.join(dst_root, subset_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in subset_images:
                shutil.copy(img_path, dest_dir)

        logger.info(
            f"  '{class_name}': "
            f"{len(subsets['train'])} train | "
            f"{len(subsets['val'])} val | "
            f"{len(subsets['test'])} test  (total: {n_total})"
        )

    logger.info(f"Dataset split complete → '{dst_root}'")
    return dst_root


# =============================================================================
# STEP 3: AUGMENTATION LAYERS (training only)
# =============================================================================

def build_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.056),
        tf.keras.layers.RandomZoom(height_factor=0.2),
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="augmentation")


# =============================================================================
# STEP 4: BUILD tf.data PIPELINES
# =============================================================================

def build_datasets(config: dict, split_dir: str):
    img_size   = config["img_size"]
    batch_size = config["batch_size"]
    seed       = config["seed"]

    AUTOTUNE = tf.data.AUTOTUNE

    def load_split(subset: str, shuffle: bool):
        target_dir = os.path.join(split_dir, subset)
        logger.info(f"Loading {subset} dataset from: {target_dir}")
        
        return tf.keras.utils.image_dataset_from_directory(
            directory=target_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=shuffle,
            seed=seed,
        )

    raw_train = load_split("train", shuffle=True)
    raw_val   = load_split("val",   shuffle=False)
    raw_test  = load_split("test",  shuffle=False)

    class_names = raw_train.class_names

    augmentation  = build_augmentation_layer()

    def preprocess_train(images, labels):
        images = augmentation(images, training=True)
        return images, labels

    def preprocess_eval(images, labels):
        return images, labels

    train_ds = raw_train.map(preprocess_train, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds   = raw_val.map(preprocess_eval, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds  = raw_test.map(preprocess_eval, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# =============================================================================
# STEP 5: PRINT DATASET SUMMARY
# =============================================================================

def print_summary(train_ds, val_ds, test_ds, class_names: list, config: dict):
    def count_samples(ds):
        return sum(labels.shape[0] for _, labels in ds)

    n_train = count_samples(train_ds)
    n_val   = count_samples(val_ds)
    n_test  = count_samples(test_ds)

    print("\n" + "=" * 55)
    print("        DATASET SUMMARY")
    print("=" * 55)
    print(f"  Number of classes : {len(class_names)}")
    print(f"  Image size        : {config['img_size']}")
    print(f"  Batch size        : {config['batch_size']}")
    print("-" * 55)
    print(f"  Training samples  : {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Test samples      : {n_test}")
    print(f"  Total samples     : {n_train + n_val + n_test}")
    print("-" * 55)
    print("  Class Indices:")
    for idx, name in enumerate(class_names):
        print(f"    [{idx}] {name}")
    print("=" * 55 + "\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess_pipeline(config: dict):
    logger.info("Starting preprocessing pipeline...")

    split_dir = split_dataset(config)
    train_ds, val_ds, test_ds, class_names = build_datasets(config, split_dir)
    print_summary(train_ds, val_ds, test_ds, class_names, config)

    logger.info("Preprocessing complete. Datasets are ready.")
    return train_ds, val_ds, test_ds, class_names


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = preprocess_pipeline(CONFIG)

    # Sanity check — inspect one batch
    for images, labels in train_ds.take(1):
        print(f"Batch shape  — images: {images.shape}, labels: {labels.shape}")
        print(f"Pixel range  — min: {images.numpy().min():.3f}, max: {images.numpy().max():.3f}")
        print(f"Classes      : {class_names}")