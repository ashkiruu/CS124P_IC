# =============================================================================
# Waste Classification - Transfer Learning with MobileNetV2
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Import our preprocessing pipeline from the previous module
from data_preprocessing import preprocess_pipeline, CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    "epochs_frozen":    10,        # Epochs with base model frozen
    "epochs_finetune":  10,        # Epochs for fine-tuning (unfrozen layers)
    "learning_rate":    1e-3,      # Initial learning rate (frozen phase)
    "finetune_lr":      1e-5,      # Lower LR for fine-tuning (very important)
    "dropout_rate":     0.3,       # Dropout to reduce overfitting
    "dense_units":      128,       # Neurons in custom dense layer
    "num_classes":      10,        # Number of waste categories
    "model_save_path":  "waste_classifier_mobilenetv2.keras",
}


# =============================================================================
# STEP 1: COMPUTE CLASS WEIGHTS
# Handles imbalanced dataset (e.g. 'trash'=453 vs 'clothes'=1892 images)
# Without this, the model ignores minority classes and just predicts
# the majority class to get a deceptively high accuracy.
# =============================================================================

def compute_weights(train_ds, class_names: list) -> dict:
    """
    Iterates through the training dataset once to collect all labels,
    then computes per-class weights inversely proportional to frequency.
    Rare classes (e.g. trash) get higher weights → model penalized more
    for misclassifying them.
    """
    print("\nComputing class weights...")

    all_labels = []
    for _, labels in train_ds:
        # labels are one-hot encoded → argmax converts back to integer index
        all_labels.extend(np.argmax(labels.numpy(), axis=1))

    all_labels  = np.array(all_labels)
    class_indices = np.unique(all_labels)

    weights = compute_class_weight(
        class_weight="balanced",   # sklearn balances weights automatically
        classes=class_indices,
        y=all_labels
    )

    class_weight_dict = dict(zip(class_indices, weights))

    print("  Class Weights:")
    for idx, name in enumerate(class_names):
        print(f"    [{idx}] {name:<12} → weight: {class_weight_dict[idx]:.4f}")

    return class_weight_dict


# =============================================================================
# STEP 2: BUILD THE MODEL
# Transfer Learning concept:
#   MobileNetV2 was trained on ImageNet (1.4M images, 1000 classes).
#   It has already learned to detect edges, textures, shapes, and objects.
#   Instead of learning all that from scratch, we REUSE those weights
#   and only train a new classification head for our 10 waste classes.
#
#   Think of it like hiring an expert photographer to sort your photos —
#   they already know what things look like, you just tell them your categories.
# =============================================================================

def build_model(config: dict) -> tf.keras.Model:
    """
    Constructs the full model:
        MobileNetV2 base (frozen, pretrained on ImageNet)
        + custom classification head (trainable)
    """

    # ------------------------------------------------------------------
    # Load MobileNetV2 base — pretrained on ImageNet
    # include_top=False  → removes the original 1000-class output layer
    #                      so we can attach our own head for 10 classes
    # weights='imagenet' → loads weights trained on ImageNet dataset
    # ------------------------------------------------------------------
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # ------------------------------------------------------------------
    # FREEZE the base model
    # WHY: We don't want to destroy the pretrained ImageNet weights
    #      during early training. The base already knows how to detect
    #      features — we only want to train our new classification head.
    #
    # Freezing means: base_model layers have trainable=False
    #                 → their weights are NOT updated during backprop
    # ------------------------------------------------------------------
    base_model.trainable = False
    print(f"\nBase model loaded: {len(base_model.layers)} layers (all frozen)")

    # ------------------------------------------------------------------
    # Build the full model using Functional API
    # We use Functional (not Sequential) because it's more explicit
    # and handles complex architectures better for transfer learning
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")

    # ------------------------------------------------------------------
    # Preprocess input for MobileNetV2
    # MobileNetV2 expects pixels in [-1, 1] range
    # Our pipeline outputs [0, 1] → this layer rescales correctly
    # ------------------------------------------------------------------
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # ------------------------------------------------------------------
    # Pass through frozen MobileNetV2 base
    # training=False → keeps BatchNormalization in inference mode
    #                  even when we later unfreeze layers
    # ------------------------------------------------------------------
    x = base_model(x, training=False)

    # ------------------------------------------------------------------
    # GlobalAveragePooling2D
    # WHY: MobileNetV2 outputs a 3D feature map (7×7×1280)
    #      GAP averages each feature map → collapses to (1280,) vector
    #      This is better than Flatten() because:
    #        - Far fewer parameters
    #        - Less overfitting
    #        - Spatially invariant (works regardless of object position)
    # ------------------------------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ------------------------------------------------------------------
    # Dense(128) — custom feature transformation
    # WHY: Learns task-specific combinations of MobileNetV2 features
    #      128 units is a good balance between capacity and overfitting
    # ------------------------------------------------------------------
    x = tf.keras.layers.Dense(
        config["dense_units"],
        activation="relu",
        name="dense_head"
    )(x)

    # ------------------------------------------------------------------
    # Dropout(0.3)
    # WHY: Randomly zeros 30% of neurons during training
    #      Forces the network to not rely on any single neuron
    #      → reduces overfitting, improves generalization
    # ------------------------------------------------------------------
    x = tf.keras.layers.Dropout(config["dropout_rate"], name="dropout")(x)

    # ------------------------------------------------------------------
    # Output layer — Dense(10, softmax)
    # WHY: 10 neurons = 1 per waste class
    #      softmax converts raw scores to probabilities that sum to 1.0
    #      e.g. [0.02, 0.01, 0.85, ...] → model is 85% confident it's cardboard
    # ------------------------------------------------------------------
    outputs = tf.keras.layers.Dense(
        config["num_classes"],
        activation="softmax",
        name="output_layer"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="WasteClassifier_MobileNetV2")
    return model, base_model


# =============================================================================
# STEP 3: COMPILE THE MODEL
# =============================================================================

def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """
    Compiles the model with optimizer, loss, and metrics.

    categorical_crossentropy → correct loss for one-hot encoded multi-class
    adam                     → adaptive learning rate, good default optimizer
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =============================================================================
# STEP 4: CALLBACKS
# These run automatically at the end of each epoch to improve training
# =============================================================================

def build_callbacks(config: dict) -> list:
    """
    EarlyStopping  → stops training when val_loss stops improving
                     prevents wasted computation and overfitting

    ReduceLROnPlateau → halves learning rate when val_loss plateaus
                        helps escape local minima

    ModelCheckpoint → saves the best model weights automatically
                      so even if training degrades later, best is kept
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,             # Stop after 4 epochs of no improvement
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,             # Multiply LR by 0.5 on plateau
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["model_save_path"],
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


# =============================================================================
# STEP 5: TRAIN - PHASE 1 (Frozen base)
# Only the classification head is trained here.
# Fast training, stable because base weights don't change.
# =============================================================================

def train_frozen_phase(model, train_ds, val_ds, class_weights, config):
    """
    Phase 1: Train only the custom head with the base frozen.
    This is fast and safe — we're just learning to map MobileNetV2
    features to our 10 waste classes.
    """
    print("\n" + "="*55)
    print("  PHASE 1: Training classification head (base frozen)")
    print("="*55)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs_frozen"],
        class_weight=class_weights,
        callbacks=build_callbacks(config),
        verbose=1,
    )
    return history


# =============================================================================
# STEP 6: TRAIN - PHASE 2 (Fine-tuning)
# Unfreeze the top layers of MobileNetV2 and retrain with a very low LR.
# WHY: After the head is trained, we can slightly adjust the base model's
#      higher-level features to better fit waste images specifically.
# CAUTION: Use a very low learning rate (1e-5) to avoid destroying
#           the pretrained weights we're building on.
# =============================================================================

def train_finetune_phase(model, base_model, train_ds, val_ds, class_weights, config):
    """
    Phase 2: Unfreeze top 30 layers of MobileNetV2 for fine-tuning.
    Fine-tuning adapts pretrained features specifically to waste images.
    """
    print("\n" + "="*55)
    print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("="*55)

    # Unfreeze the entire base model first
    base_model.trainable = True

    # Re-freeze everything EXCEPT the last 30 layers
    # WHY: Early layers detect basic features (edges, colors) — universal
    #      Later layers detect complex patterns — more task-specific
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Unfrozen layers: {trainable_count} / {len(base_model.layers)}")

    # MUST recompile after changing trainable layers
    # Use much lower LR to make small, careful adjustments
    model = compile_model(model, learning_rate=config["finetune_lr"])

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs_finetune"],
        class_weight=class_weights,
        callbacks=build_callbacks(config),
        verbose=1,
    )
    return history_ft


# =============================================================================
# STEP 7: PLOT TRAINING HISTORY
# =============================================================================

def plot_history(history_frozen, history_finetune=None):
    """
    Plots accuracy and loss curves for both training phases.
    Helps visually diagnose overfitting, underfitting, or instability.
    """
    # Combine both phases into one continuous history
    acc     = history_frozen.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"]
    loss    = history_frozen.history["loss"]
    val_loss= history_frozen.history["val_loss"]

    if history_finetune:
        acc      += history_finetune.history["accuracy"]
        val_acc  += history_finetune.history["val_accuracy"]
        loss     += history_finetune.history["loss"]
        val_loss += history_finetune.history["val_loss"]

    epochs_range = range(len(acc))
    frozen_end   = len(history_frozen.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MobileNetV2 Transfer Learning — Training History", fontsize=14)

    # --- Accuracy plot ---
    axes[0].plot(epochs_range, acc,     label="Train Accuracy",      color="royalblue")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy",  color="orange")
    axes[0].axvline(x=frozen_end - 1, color="red", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss plot ---
    axes[1].plot(epochs_range, loss,     label="Train Loss",      color="royalblue")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss",  color="orange")
    axes[1].axvline(x=frozen_end - 1, color="red", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Plot saved as training_history.png")


# =============================================================================
# STEP 8: EVALUATE ON TEST SET
# Done ONCE at the very end — this is the honest final score
# =============================================================================

def evaluate_model(model, test_ds):
    """
    Evaluates on the held-out test set.
    This number is the one you report — it reflects real-world performance.
    """
    print("\n" + "="*55)
    print("  FINAL EVALUATION ON TEST SET")
    print("="*55)

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    return test_loss, test_acc


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def training_pipeline():

    # --- Load preprocessed datasets from previous module ---
    train_ds, val_ds, test_ds, class_names = preprocess_pipeline(CONFIG)

    # --- Class weights for imbalanced dataset ---
    class_weights = compute_weights(train_ds, class_names)

    # --- Build model ---
    model, base_model = build_model(TRAIN_CONFIG)
    model = compile_model(model, TRAIN_CONFIG["learning_rate"])
    model.summary()

    # --- Phase 1: Train head only ---
    history_frozen = train_frozen_phase(
        model, train_ds, val_ds, class_weights, TRAIN_CONFIG
    )

    # --- Phase 2: Fine-tune top layers ---
    history_finetune = train_finetune_phase(
        model, base_model, train_ds, val_ds, class_weights, TRAIN_CONFIG
    )

    # --- Plot results ---
    plot_history(history_frozen, history_finetune)

    # --- Final evaluation ---
    evaluate_model(model, test_ds)

    # --- Save model ---
    model.save(TRAIN_CONFIG["model_save_path"])
    print(f"\nModel saved → {TRAIN_CONFIG['model_save_path']}")

    return model


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    model = training_pipeline()