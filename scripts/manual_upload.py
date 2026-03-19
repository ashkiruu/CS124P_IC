# =============================================================================
# Waste Classification - Inference / Prediction Script
# Model: MobileNetV2 | Input: (224, 224, 3) | Pixel range: [0, 255]
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "waste_classifier_mobilenetv2.keras"

# Class names MUST match the exact order used during training.
# This order comes from sorted() on folder names in dataset_split/train/
# which is alphabetical — do not rearrange this list.
CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]

# Confidence threshold — predictions below this are flagged as uncertain
CONFIDENCE_THRESHOLD = 0.60


# =============================================================================
# STEP 1: LOAD MODEL
# =============================================================================

def load_model(model_path: str) -> tf.keras.Model:
    """
    Loads the trained Keras model from disk.
    compile=False → we only need inference, not training,
    so we skip reloading optimizer and loss (faster loading).
    """
    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.\n")
    return model


# =============================================================================
# STEP 2: IMAGE PREPROCESSING
# =============================================================================

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and prepares a single image for inference.

    Steps:
        1. Open image with PIL
        2. Convert to RGB — ensures 3 channels always
           (some images are RGBA=4 channels or grayscale=1 channel)
        3. Resize to (224, 224) — MobileNetV2 required input size
        4. Convert to NumPy float32 array — TensorFlow needs float, not uint8
        5. Expand dims to (1, 224, 224, 3) — model expects a BATCH, not
           a single image. Adding dimension at axis=0 creates a batch of 1.

    WHY WE DO NOT NORMALIZE (no division by 255):
        Our model's first internal step is:
            mobilenet_v2.preprocess_input()
        which expects raw pixel values in [0, 255] and converts them
        to [-1, 1] internally. If we divide by 255 here, the model
        receives [0, 1] and then rescales to [-0.996, -0.992] — completely
        wrong range → garbage predictions. Let the model handle it.
    """
    # Open and convert to RGB (handles PNG with alpha, grayscale, etc.)
    img = Image.open(image_path).convert("RGB")

    # Resize to exactly what MobileNetV2 expects
    img = img.resize((224, 224))

    # Convert PIL image → NumPy array, cast to float32
    # Shape at this point: (224, 224, 3)
    img_array = np.array(img, dtype=np.float32)

    # WHY EXPAND DIMS:
    # model.predict() expects shape (batch_size, height, width, channels)
    # A single image has shape (224, 224, 3) — missing the batch dimension
    # np.expand_dims adds it: (224, 224, 3) → (1, 224, 224, 3)
    # This tells the model: "here is a batch of 1 image"
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =============================================================================
# STEP 3: PREDICTION
# =============================================================================

def predict_image(model: tf.keras.Model, image_path: str) -> dict:
    """
    Runs inference on a single image and returns structured results.

    HOW SOFTMAX OUTPUT WORKS:
        The model's final layer is Dense(10, activation='softmax').
        Softmax converts raw scores (logits) into probabilities:
            - Each value is between 0.0 and 1.0
            - All 10 values sum to exactly 1.0
            - Higher value = model is more confident about that class

        Example output for one image:
            [0.01, 0.00, 0.02, 0.01, 0.05, 0.03, 0.02, 0.83, 0.02, 0.01]
             bat   bio   card  cloth glass metal paper PLST  shoe  trash
                                                        ↑ 83% confident = plastic

        np.argmax() finds the index with the highest probability → class index
        CLASS_NAMES[index] maps that index back to the human-readable name.

    Returns a dict with:
        top1_class      : predicted class name
        top1_confidence : confidence as percentage (0-100)
        top3            : list of (class_name, confidence%) sorted desc
        is_uncertain    : True if confidence < CONFIDENCE_THRESHOLD
        probabilities   : full array of all 10 class probabilities
    """
    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Run inference — model.predict() handles the preprocess_input internally
    # Output shape: (1, 10) — probabilities for 1 image across 10 classes
    predictions = model.predict(img_array, verbose=0)

    # Extract the probability array for our single image
    # Shape goes from (1, 10) → (10,)
    probabilities = predictions[0]

    # Top-1: highest probability class
    top1_idx        = np.argmax(probabilities)
    top1_class      = CLASS_NAMES[top1_idx]
    top1_confidence = float(probabilities[top1_idx]) * 100

    # Top-3: sort all classes by probability descending, take first 3
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [
        (CLASS_NAMES[i], float(probabilities[i]) * 100)
        for i in top3_indices
    ]

    # Flag low-confidence predictions
    is_uncertain = top1_confidence < (CONFIDENCE_THRESHOLD * 100)

    return {
        "top1_class":      top1_class,
        "top1_confidence": top1_confidence,
        "top3":            top3,
        "is_uncertain":    is_uncertain,
        "probabilities":   probabilities,
    }


# =============================================================================
# STEP 4: PRINT RESULTS
# =============================================================================

def print_results(result: dict, image_path: str):
    """
    Prints prediction results in a clean, readable format.
    Includes a warning if the model is not confident enough.
    """
    print("=" * 50)
    print("  PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Image     : {image_path}")
    print(f"  Predicted : {result['top1_class'].upper()}")
    print(f"  Confidence: {result['top1_confidence']:.2f}%")

    if result["is_uncertain"]:
        print(f"\n  ⚠️  Low confidence — model is uncertain.")
        print(f"  Consider using a clearer image.")

    print(f"\n  Top 3 Predictions:")
    print(f"  {'Rank':<6} {'Class':<14} {'Confidence':>10}")
    print(f"  {'-'*34}")
    for rank, (cls, conf) in enumerate(result["top3"], start=1):
        bar   = "█" * int(conf / 5)   # visual confidence bar
        print(f"  #{rank:<5} {cls:<14} {conf:>8.2f}%  {bar}")

    print("=" * 50 + "\n")


# =============================================================================
# STEP 5: DISPLAY IMAGE WITH PREDICTION
# =============================================================================

def display_prediction(image_path: str, result: dict):
    """
    Shows the input image with the prediction overlaid as the title.
    Color-codes the title: green = confident, red = uncertain.
    """
    img = Image.open(image_path).convert("RGB")

    title_color = "red" if result["is_uncertain"] else "green"
    title = (
        f"Predicted: {result['top1_class'].upper()} "
        f"({result['top1_confidence']:.2f}%)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=title_color)

    # --- Left: the image ---
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=11)

    # --- Right: bar chart of top-3 probabilities ---
    classes = [cls for cls, _ in result["top3"]]
    confs   = [conf for _, conf in result["top3"]]
    colors  = ["#2ecc71" if cls == result["top1_class"]
               else "#3498db" for cls in classes]

    bars = axes[1].barh(classes[::-1], confs[::-1], color=colors[::-1])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)", fontsize=11)
    axes[1].set_title("Top 3 Predictions", fontsize=11)

    # Add percentage labels on bars
    for bar, conf in zip(bars, confs[::-1]):
        axes[1].text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center", fontsize=10
        )

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Result saved as prediction_result.png")


# =============================================================================
# STEP 6: BATCH PREDICTION (bonus — test multiple images at once)
# =============================================================================

def predict_batch(model: tf.keras.Model, image_paths: list):
    """
    Runs prediction on a list of image paths.
    Useful for testing multiple images in one go.
    """
    print(f"\nRunning batch prediction on {len(image_paths)} images...\n")
    results = []

    for path in image_paths:
        try:
            result = predict_image(model, path)
            print(f"  {path:<30} → {result['top1_class']:<14} ({result['top1_confidence']:.2f}%)")
            results.append((path, result))
        except FileNotFoundError:
            print(f"  {path:<30} → ❌ File not found, skipping.")
        except Exception as e:
            print(f"  {path:<30} → ❌ Error: {e}")

    return results


# =============================================================================
# MAIN — test with a sample image
# =============================================================================

def main():
    """
    Entry point. Change IMAGE_PATH to point to any image you want to classify.
    """
    IMAGE_PATH = r"manual_dataset\battery.jpg"   # ← change this to your image path

    # --- Load model once (expensive — do this once, reuse for many predictions) ---
    model = load_model(MODEL_PATH)

    # --- Single image prediction ---
    try:
        result = predict_image(model, IMAGE_PATH)
        print_results(result, IMAGE_PATH)
        display_prediction(IMAGE_PATH, result)

    except FileNotFoundError:
        print(f"Image not found: '{IMAGE_PATH}'")
        print("Update IMAGE_PATH in main() to point to a real image file.")
        return

    test_images = [
            r"manual_dataset\battery.jpg",
            r"manual_dataset\cardboard.jpg",
            r"manual_dataset\clothes.jpg",
            r"manual_dataset\food.jpg",
            r"manual_dataset\glass.jpg",
            r"manual_dataset\metal.jpg",
            r"manual_dataset\paper.jpg",
            r"manual_dataset\plastic.jpg",
            r"manual_dataset\shoes.jpeg",    # ← .jpeg not .jpg, keep as-is
            r"manual_dataset\trash.jpg",
        ]
        
    predict_batch(model, test_images)


if __name__ == "__main__":
    main()