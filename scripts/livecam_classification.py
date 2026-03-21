# =============================================================================
# Waste Classification - Real-Time Webcam Inference
# Model: MobileNetV2 | Input: (224, 224, 3) | Pixel range: [0, 255]
# =============================================================================

import cv2
import numpy as np
import tensorflow as tf
from collections import deque


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH   = "waste_classifier_mobilenetv2.keras"
CLASS_NAMES  = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]

# UI settings
CONFIDENCE_THRESHOLD = 0.60       # below this → show UNCERTAIN
PREDICTION_SMOOTHING = 10         # number of frames to average over
PREDICT_EVERY_N_FRAMES = 3        # run model every N frames (CPU optimization)

# Colors (BGR format for OpenCV)
COLOR_CONFIDENT   = (0, 220, 0)    # green
COLOR_UNCERTAIN   = (0, 100, 255)  # orange
COLOR_BACKGROUND  = (30, 30, 30)   # dark gray
COLOR_WHITE       = (255, 255, 255)
COLOR_BAR_BG      = (80, 80, 80)
COLOR_BAR_FILL    = (0, 200, 100)


# =============================================================================
# STEP 1: LOAD MODEL
# =============================================================================

def load_model(model_path: str) -> tf.keras.Model:
    """
    Loads the trained Keras model.
    compile=False → inference only, skips loading optimizer state.
    """
    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded. Starting webcam...\n")
    return model


# =============================================================================
# STEP 2: PREPROCESS A SINGLE FRAME
# =============================================================================

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Converts a raw OpenCV webcam frame into model-ready input.

    WHY BGR → RGB:
        OpenCV reads frames in BGR channel order (Blue, Green, Red).
        This is a historical quirk of OpenCV's design.
        Our model was trained on RGB images (PIL loads as RGB).
        If we skip this conversion, red and blue channels are swapped
        → colors look wrong to the model → predictions are unreliable.

    WHY NO DIVISION BY 255:
        mobilenet_v2.preprocess_input() inside the model expects
        raw pixel values in [0, 255] and converts them to [-1, 1].
        Dividing by 255 first gives [0, 1] → model converts to
        roughly [-0.996, -0.992] → completely wrong range → garbage output.

    WHY EXPAND DIMS:
        model.predict() expects (batch, H, W, C).
        One frame has shape (224, 224, 3) → we add batch dim
        → (1, 224, 224, 3) = "a batch of 1 frame".
    """
    # OpenCV BGR → RGB (matches training data format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to MobileNetV2's required input size
    resized = cv2.resize(rgb_frame, (224, 224))

    # Convert to float32 — TF requires float, not uint8
    # DO NOT divide by 255 — model handles normalization internally
    img_array = np.array(resized, dtype=np.float32)

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


# =============================================================================
# STEP 3: RUN PREDICTION
# =============================================================================

def predict_frame(model: tf.keras.Model, preprocessed: np.ndarray) -> tuple:
    """
    Runs inference on one preprocessed frame.

    Returns:
        top1_class      : predicted class name string
        top1_confidence : float in [0, 1]
        top3            : list of (class_name, confidence) tuples
        probabilities   : full array of 10 probabilities
    """
    # model() is faster than model.predict() for single frames
    # because predict() has overhead for batching logic
    predictions   = model(preprocessed, training=False)
    probabilities = predictions.numpy()[0]    # shape (10,)

    # Top-1
    top1_idx        = np.argmax(probabilities)
    top1_class      = CLASS_NAMES[top1_idx]
    top1_confidence = float(probabilities[top1_idx])

    # Top-3 sorted descending
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(probabilities[i])) for i in top3_indices]

    return top1_class, top1_confidence, top3, probabilities


# =============================================================================
# STEP 4: PREDICTION SMOOTHER
# Averages probabilities across the last N frames to stabilize output.
# Without this, predictions flicker every frame on CPU.
# =============================================================================

class PredictionSmoother:
    """
    Maintains a rolling window of probability arrays.
    Averages them to produce a stable, smoothed prediction.

    Example: if last 10 frames said:
        8x plastic (avg prob 0.75)
        2x glass   (avg prob 0.18)
    → smoothed output: plastic 75%, glass 18% (stable, no flicker)
    """
    def __init__(self, window_size: int, num_classes: int):
        self.window = deque(maxlen=window_size)
        self.num_classes = num_classes

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        self.window.append(probabilities)
        return np.mean(self.window, axis=0)   # average across all frames in window

    def is_ready(self) -> bool:
        return len(self.window) >= 3          # need at least 3 frames to smooth


# =============================================================================
# STEP 5: DRAW OVERLAY UI ON FRAME
# =============================================================================

def draw_overlay(frame: np.ndarray, top1_class: str,
                 top1_confidence: float, top3: list) -> np.ndarray:
    """
    Draws the prediction overlay directly onto the video frame.

    Layout:
        Top bar     → predicted class + confidence
        Bottom panel→ top-3 predictions with mini confidence bars
        Warning     → UNCERTAIN banner if confidence < threshold
    """
    h, w = frame.shape[:2]
    is_uncertain = top1_confidence < CONFIDENCE_THRESHOLD
    label_color  = COLOR_UNCERTAIN if is_uncertain else COLOR_CONFIDENT

    # ------------------------------------------------------------------
    # TOP BAR — main prediction
    # ------------------------------------------------------------------
    cv2.rectangle(frame, (0, 0), (w, 60), COLOR_BACKGROUND, -1)

    main_label = (
        f"UNCERTAIN ({top1_confidence*100:.1f}%)"
        if is_uncertain
        else f"{top1_class.upper()}  {top1_confidence*100:.1f}%"
    )

    cv2.putText(
        frame, main_label,
        (15, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2, label_color, 2, cv2.LINE_AA
    )

    # ------------------------------------------------------------------
    # BOTTOM PANEL — top-3 predictions with confidence bars
    # ------------------------------------------------------------------
    panel_h = 120
    panel_y = h - panel_h
    cv2.rectangle(frame, (0, panel_y), (w, h), COLOR_BACKGROUND, -1)

    cv2.putText(
        frame, "Top 3 Predictions:",
        (15, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, COLOR_WHITE, 1, cv2.LINE_AA
    )

    for i, (cls_name, conf) in enumerate(top3):
        y_pos    = panel_y + 45 + i * 26
        bar_x    = 200
        bar_maxw = w - bar_x - 20
        bar_w    = int(bar_maxw * conf)

        # Class name + percentage text
        label = f"#{i+1} {cls_name:<14} {conf*100:5.1f}%"
        cv2.putText(
            frame, label,
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_CONFIDENT if i == 0 else COLOR_WHITE,
            1, cv2.LINE_AA
        )

        # Confidence bar background
        cv2.rectangle(frame, (bar_x, y_pos - 12),
                      (bar_x + bar_maxw, y_pos + 3), COLOR_BAR_BG, -1)

        # Confidence bar fill
        if bar_w > 0:
            bar_color = COLOR_BAR_FILL if i == 0 else (100, 160, 100)
            cv2.rectangle(frame, (bar_x, y_pos - 12),
                          (bar_x + bar_w, y_pos + 3), bar_color, -1)

    # ------------------------------------------------------------------
    # UNCERTAIN WARNING BANNER (flashes in center)
    # ------------------------------------------------------------------
    if is_uncertain:
        cv2.rectangle(frame, (w//2 - 130, h//2 - 30),
                      (w//2 + 130, h//2 + 30), (0, 50, 180), -1)
        cv2.putText(
            frame, "LOW CONFIDENCE",
            (w//2 - 115, h//2 + 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85, COLOR_WHITE, 2, cv2.LINE_AA
        )

    # ------------------------------------------------------------------
    # CORNER INFO — controls reminder
    # ------------------------------------------------------------------
    cv2.putText(
        frame, "Press Q to quit",
        (w - 165, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (160, 160, 160), 1, cv2.LINE_AA
    )

    return frame


# =============================================================================
# STEP 6: MAIN REAL-TIME LOOP
# =============================================================================

def run_webcam(model: tf.keras.Model):
    """
    Main inference loop:
        1. Capture frame from webcam
        2. Every N frames: preprocess + predict + update smoother
        3. Every frame: draw cached prediction onto frame
        4. Display frame
        5. Exit on 'q'

    WHY PREDICT EVERY N FRAMES (not every frame):
        On CPU, model inference takes ~200-500ms per frame.
        If we predict every frame, the video becomes a slideshow.
        By predicting every 3 frames and DISPLAYING every frame,
        we get smooth video (30fps display) with periodic predictions.
        The smoother fills in the gaps between predictions.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1) if you have multiple cameras.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother     = PredictionSmoother(PREDICTION_SMOOTHING, len(CLASS_NAMES))
    frame_count  = 0

    # Cache last prediction — displayed every frame even when not recomputing
    last_top1_class = "initializing..."
    last_confidence = 0.0
    last_top3       = [(c, 0.0) for c in CLASS_NAMES[:3]]

    print("Webcam started. Hold waste items in front of the camera.")
    print("Press 'Q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        frame_count += 1

        # ------------------------------------------------------------------
        # Run prediction every N frames (CPU optimization)
        # ------------------------------------------------------------------
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            preprocessed           = preprocess_frame(frame)
            top1_class, top1_conf, top3, probs = predict_frame(model, preprocessed)
            smoothed_probs         = smoother.update(probs)

            # Use smoothed probabilities for final output
            smoothed_top1_idx      = np.argmax(smoothed_probs)
            last_top1_class        = CLASS_NAMES[smoothed_top1_idx]
            last_confidence        = float(smoothed_probs[smoothed_top1_idx])
            top3_indices           = np.argsort(smoothed_probs)[::-1][:3]
            last_top3              = [
                (CLASS_NAMES[i], float(smoothed_probs[i]))
                for i in top3_indices
            ]

        # ------------------------------------------------------------------
        # Draw overlay using cached (or freshly computed) prediction
        # ------------------------------------------------------------------
        if smoother.is_ready():
            frame = draw_overlay(frame, last_top1_class, last_confidence, last_top3)
        else:
            # Show loading message while smoother warms up
            cv2.putText(
                frame, "Warming up...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, COLOR_WHITE, 2, cv2.LINE_AA
            )

        cv2.imshow("Waste Classifier — Real-Time", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

def main():
    model = load_model(MODEL_PATH)
    run_webcam(model)


if __name__ == "__main__":
    main()
    