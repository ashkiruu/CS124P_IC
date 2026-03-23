import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, url_for
from collections import deque
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "ecoscan_secret_key"

# --- Configuration & Model Load ---
MODEL_PATH = "waste_classifier_mobilenetv2.keras"
CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# File Paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
TEST_FOLDER = os.path.join('static', 'manual_dataset')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Smoothing & Optimization
smoother_window = deque(maxlen=10)
PREDICT_EVERY_N_FRAMES = 3
frame_count = 0

# Cached results to show between model runs
last_results = {"class": "Initializing...", "conf": 0.0, "top3": []}

# =============================================================================
# PREPROCESSING HELPERS
# =============================================================================

def preprocess_frame(frame):
    """Processes a single OpenCV frame (BGR NumPy array)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    img_array = np.array(resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def preprocess_file(image_path):
    """Processes an image file from disk (Manual/Batch)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

# =============================================================================
# LIVE FEED GENERATOR
# =============================================================================

def gen_frames():
    global frame_count, last_results
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            preprocessed = preprocess_frame(frame)
            preds = model(preprocessed, training=False).numpy()[0]
            smoother_window.append(preds)
            
            if len(smoother_window) >= 3:
                avg_preds = np.mean(smoother_window, axis=0)
                top_idx = np.argmax(avg_preds)
                top3_indices = np.argsort(avg_preds)[::-1][:3]
                
                last_results = {
                    "class": CLASS_NAMES[top_idx],
                    "conf": float(avg_preds[top_idx]),
                    "top3": [(CLASS_NAMES[i], float(avg_preds[i])) for i in top3_indices]
                }

        color = (0, 255, 0) if last_results["conf"] > 0.6 else (0, 165, 255)
        cv2.putText(frame, f"{last_results['class'].upper()} ({last_results['conf']*100:.1f}%)", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/live')
def live_mode():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Streaming route for live.html. Required to fix BuildError."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    """API endpoint for live.html JavaScript updates."""
    return jsonify(last_results)

@app.route('/manual', methods=['GET', 'POST'])
def manual_mode():
    result = None
    image_url = None
    batch_results = None
    
    if request.method == 'POST':
        # --- Handle Batch Test ---
        if 'run_batch' in request.form:
            test_images = [
                "battery.jpg", "cardboard.jpg", "clothes.jpg",
                "glass.jpg", "metal.jpg", "paper.jpg",
                "plastic.jpg", "shoes.jpeg", "trash.jpg"
            ]
            batch_results = []
            for img_name in test_images:
                path = os.path.join(TEST_FOLDER, img_name)
                if os.path.exists(path):
                    img_arr = preprocess_file(path)
                    preds = model.predict(img_arr, verbose=0)[0]
                    top_idx = np.argmax(preds)
                    batch_results.append({
                        "filename": img_name,
                        "top1_class": CLASS_NAMES[top_idx],
                        "top1_confidence": round(float(preds[top_idx]) * 100, 2),
                        "url": url_for('static', filename=f'manual_dataset/{img_name}'),
                        "is_uncertain": float(preds[top_idx]) < 0.60
                    })
            return render_template('manual.html', batch_results=batch_results)

        # --- Handle User Upload ---
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                img_arr = preprocess_file(filepath)
                preds = model.predict(img_arr, verbose=0)[0]
                top1_idx = np.argmax(preds)
                top3_indices = np.argsort(preds)[::-1][:3]
                
                result = {
                    "top1_class": CLASS_NAMES[top1_idx],
                    "top1_confidence": round(float(preds[top1_idx]) * 100, 2),
                    "top3": [(CLASS_NAMES[i], round(float(preds[i]) * 100, 2)) for i in top3_indices],
                    "is_uncertain": float(preds[top1_idx]) < 0.60
                }
                image_url = url_for('static', filename=f'uploads/{filename}')
    
    return render_template('manual.html', result=result, image_url=image_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)