# =============================================================================
# Waste Classification - Model Evaluation
# Generates classification report + confusion matrix
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import preprocess_pipeline, CONFIG


# =============================================================================
# STEP 1: LOAD THE SAVED MODEL
# =============================================================================

def load_model(model_path: str) -> tf.keras.Model:
    """
    Loads the trained Keras model from disk.
    We use compile=False because we are only doing inference,
    not training — so we don't need optimizer/loss reloaded.
    """
    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully.\n")
    return model


# =============================================================================
# STEP 2: EXTRACT TRUE LABELS AND PREDICTIONS FROM TEST DATASET
#
# IMPORTANT — Why we do this carefully:
#   test_ds was built with shuffle=False, so image order is preserved.
#   We iterate through every batch, collecting:
#     - true labels  (from the dataset, one-hot encoded → converted to int)
#     - predictions  (model output softmax → argmax → class index)
#   Both lists stay aligned because we process them batch by batch together.
# =============================================================================

def get_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset):
    """
    Runs inference on the entire test dataset.
    Returns:
        y_true : list of true class indices
        y_pred : list of predicted class indices
    """
    print("Running predictions on test dataset...")

    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        # model.predict on a batch → shape (batch_size, 10) softmax probabilities
        predictions = model(batch_images, training=False)

        # argmax converts:
        #   one-hot [0,0,1,0,...] → integer index 2  (true label)
        #   softmax [0.02,0.91,...] → integer index 1  (predicted label)
        true_indices = np.argmax(batch_labels.numpy(), axis=1)
        pred_indices = np.argmax(predictions.numpy(), axis=1)

        y_true.extend(true_indices)
        y_pred.extend(pred_indices)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"Total samples evaluated : {len(y_true)}")
    print(f"Overall accuracy        : {np.mean(y_true == y_pred) * 100:.2f}%\n")

    return y_true, y_pred


# =============================================================================
# STEP 3: CLASSIFICATION REPORT
# Shows per-class precision, recall, F1-score, and support
# =============================================================================

def print_classification_report(y_true, y_pred, class_names: list):
    """
    Prints a detailed per-class performance breakdown.
    'support' = number of actual test samples for that class.
    """
    print("=" * 65)
    print("  CLASSIFICATION REPORT")
    print("=" * 65)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4              # 4 decimal places for precision
    )
    print(report)
    return report


# =============================================================================
# STEP 4: CONFUSION MATRIX — RAW + NORMALIZED
# =============================================================================

def compute_confusion_matrix(y_true, y_pred, class_names: list):
    """
    Computes both:
        raw_cm  : actual counts of predictions per class pair
        norm_cm : row-normalized (shows % of true class predicted as each class)

    Row = actual class
    Col = predicted class
    Diagonal = correct predictions
    Off-diagonal = mistakes
    """
    raw_cm  = confusion_matrix(y_true, y_pred)

    # Normalize each row by its total — shows proportion, not raw count
    # Useful when classes have different numbers of test samples
    norm_cm = raw_cm.astype("float") / raw_cm.sum(axis=1, keepdims=True)

    return raw_cm, norm_cm


# =============================================================================
# STEP 5: PLOT CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(raw_cm, norm_cm, class_names: list):
    """
    Plots side-by-side:
        Left  — raw counts     (how many images per cell)
        Right — normalized %   (proportion per true class)

    Color interpretation:
        Dark diagonal   = model is correct for that class ✅
        Bright off-diag = common misclassification ❌
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        "Confusion Matrix — Waste Classification (MobileNetV2)",
        fontsize=15, fontweight="bold", y=1.01
    )

    # --- LEFT: Raw counts ---
    sns.heatmap(
        raw_cm,
        annot=True,
        fmt="d",                        # integer format
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Raw Counts", fontsize=13, pad=12)
    axes[0].set_ylabel("True Label",      fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # --- RIGHT: Normalized (proportions) ---
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".2f",                      # 2 decimal float
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0, vmax=1.0,             # fix scale to [0,1] for all classes
        cbar_kws={"shrink": 0.8}
    )
    axes[1].set_title("Normalized (Row %)", fontsize=13, pad=12)
    axes[1].set_ylabel("True Label",      fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")


# =============================================================================
# STEP 6: PER-CLASS ACCURACY SUMMARY
# Quick readable table showing how well each class performed
# =============================================================================

def print_per_class_accuracy(raw_cm, class_names: list):
    """
    Extracts diagonal values from the confusion matrix to show
    per-class accuracy — useful for spotting weak classes quickly.
    """
    print("\n" + "=" * 50)
    print("  PER-CLASS ACCURACY SUMMARY")
    print("=" * 50)
    print(f"  {'Class':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)

    for i, class_name in enumerate(class_names):
        correct = raw_cm[i, i]
        total   = raw_cm[i].sum()
        acc     = correct / total * 100
        # Flag classes below 85% accuracy for attention
        flag = " ⚠️" if acc < 85 else " ✅"
        print(f"  {class_name:<14} {correct:>8} {total:>8} {acc:>9.1f}%{flag}")

    print("=" * 50)


# =============================================================================
# STEP 7: FIND TOP MISCLASSIFICATIONS
# Tells you exactly which class pairs the model confuses most
# =============================================================================

def print_top_misclassifications(raw_cm, class_names: list, top_n: int = 5):
    """
    Scans the off-diagonal of the confusion matrix to find
    the most common misclassification pairs.

    Example output:
        plastic → glass : 34 times
        trash   → paper : 21 times
    """
    print(f"\n  TOP {top_n} MISCLASSIFICATIONS")
    print("-" * 50)

    mistakes = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue                # skip correct predictions
            count = raw_cm[true_idx, pred_idx]
            if count > 0:
                mistakes.append((count, class_names[true_idx], class_names[pred_idx]))

    # Sort by count descending
    mistakes.sort(reverse=True)

    for count, true_cls, pred_cls in mistakes[:top_n]:
        print(f"  {true_cls:<14} → {pred_cls:<14} : {count} times")

    print("=" * 50 + "\n")


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def evaluate_pipeline(model_path: str = "waste_classifier_mobilenetv2.keras"):

    # --- Load preprocessed test dataset ---
    _, _, test_ds, class_names = preprocess_pipeline(CONFIG)

    # --- Load trained model ---
    model = load_model(model_path)

    # --- Get predictions ---
    y_true, y_pred = get_predictions(model, test_ds)

    # --- Classification report ---
    print_classification_report(y_true, y_pred, class_names)

    # --- Confusion matrix ---
    raw_cm, norm_cm = compute_confusion_matrix(y_true, y_pred, class_names)

    # --- Per-class accuracy table ---
    print_per_class_accuracy(raw_cm, class_names)

    # --- Top misclassifications ---
    print_top_misclassifications(raw_cm, class_names, top_n=5)

    # --- Plot ---
    plot_confusion_matrix(raw_cm, norm_cm, class_names)

    return y_true, y_pred, raw_cm, class_names


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    y_true, y_pred, raw_cm, class_names = evaluate_pipeline()