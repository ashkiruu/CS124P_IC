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
# CONFIGURATION
# =============================================================================

CONFIG = {
    "dataset_root": r"archive\standardized_384",         # Root folder with class subfolders
    "split_output":  "dataset_split/",   # Where train/val/test folders go
    "img_size":      (224, 224),          # MobileNetV2-compatible input size
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
    Copies images from the raw dataset root into:
        dataset_split/train/<class>/
        dataset_split/val/<class>/
        dataset_split/test/<class>/

    Skips if the split directory already exists (safe to re-run).
    """
    src_root = config["dataset_root"]
    dst_root = config["split_output"]
    splits   = config["splits"]
    seed     = config["seed"]

    assert abs(sum(splits.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    if os.path.exists(dst_root):
        logger.info(f"Split directory already exists at '{dst_root}'. Skipping.")
        return dst_root

    random.seed(seed)
    class_names = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ])

    if not class_names:
        raise ValueError(f"No class subfolders found in '{src_root}'.")

    logger.info(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        images = collect_valid_images(os.path.join(src_root, class_name))
        random.shuffle(images)

        n_total = len(images)
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
# Applied inside the tf.data pipeline — runs on GPU automatically
# =============================================================================

def build_augmentation_layer():
    """
    Returns a Sequential augmentation model applied ONLY during training.
    Using Keras layers instead of ImageDataGenerator gives GPU acceleration
    and is the recommended modern approach.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.056),   # ≈ ±20 degrees (20/360)
        tf.keras.layers.RandomZoom(height_factor=0.2),  # zoom up to 20%
        tf.keras.layers.RandomFlip("horizontal"),        # horizontal flip
    ], name="augmentation")


# =============================================================================
# STEP 4: BUILD tf.data PIPELINES
# =============================================================================

def build_datasets(config: dict, split_dir: str):
    """
    Loads images from split folders using image_dataset_from_directory,
    then builds optimized tf.data pipelines for train, val, and test.

    Pipeline steps:
        1. Load & decode images (handled by image_dataset_from_directory)
        2. Normalize pixel values to [0, 1]
        3. Apply augmentation to training set only
        4. Cache, shuffle, prefetch for performance
    """
    img_size   = config["img_size"]
    batch_size = config["batch_size"]
    seed       = config["seed"]

    AUTOTUNE = tf.data.AUTOTUNE

    # ------------------------------------------------------------------
    # Load raw datasets (images are resized and batched automatically)
    # label_mode='categorical' → one-hot encoded labels (for softmax)
    # ------------------------------------------------------------------
    def load_split(subset: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(split_dir, subset),
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",   # one-hot labels for multi-class
            shuffle=shuffle,
            seed=seed,
        )

    raw_train = load_split("train", shuffle=True)
    raw_val   = load_split("val",   shuffle=False)
    raw_test  = load_split("test",  shuffle=False)

    # Capture class names before pipeline transforms strip them
    class_names = raw_train.class_names

    # ------------------------------------------------------------------
    # Normalization layer — learned from training data only
    # Rescales pixel values from [0, 255] → [0, 1]
    # NOTE: For MobileNetV2, swap this with:
    #   tf.keras.applications.mobilenet_v2.preprocess_input  (→ [-1, 1])
    # ------------------------------------------------------------------
    augmentation  = build_augmentation_layer()

    def preprocess_train(images, labels):
    # No normalization here — model handles it via preprocess_input internally
        images = augmentation(images, training=True)
        return images, labels



    def preprocess_eval(images, labels):
        # No normalization here — model handles it via preprocess_input internally
        return images, labels

    # ------------------------------------------------------------------
    # Build optimized pipelines
    # .cache()    → keeps dataset in memory after first epoch (faster)
    # .prefetch() → overlaps data loading with model training
    # ------------------------------------------------------------------
    train_ds = (
        raw_train
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        raw_val
        .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        raw_test
        .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names


# =============================================================================
# STEP 5: PRINT DATASET SUMMARY
# =============================================================================

def print_summary(train_ds, val_ds, test_ds, class_names: list, config: dict):
    """
    Prints class indices and sample counts per split.
    """
    def count_samples(ds):
        return sum(labels.shape[0] for _, labels in ds)

    n_train = count_samples(train_ds)
    n_val   = count_samples(val_ds)
    n_test  = count_samples(test_ds)

    print("\n" + "=" * 55)
    print("         DATASET SUMMARY")
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
    """
    Full preprocessing pipeline:
        1. Split raw dataset into train / val / test folders
        2. Build optimized tf.data pipelines
        3. Print summary

    Returns (train_ds, val_ds, test_ds, class_names)
    """
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