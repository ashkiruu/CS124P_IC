import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
from collections import deque

app = Flask(__name__)

# --- Configuration & Model Load ---
MODEL_PATH = "waste_classifier_mobilenetv2.keras"
CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Smoothing & Optimization
smoother_window = deque(maxlen=10)
PREDICT_EVERY_N_FRAMES = 3
frame_count = 0

# Cached results to show between model runs
last_results = {"class": "Initializing...", "conf": 0.0, "top3": []}

def preprocess_frame(frame):
    # Resize and convert BGR (OpenCV) to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    img_array = np.array(resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def gen_frames():
    global frame_count, last_results
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        
        # Inference Logic (optimized)
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

        # --- Draw UI on the frame (similar to Marcin's script but for Web) ---
        # Draw a simple overlay for the class and confidence
        color = (0, 255, 0) if last_results["conf"] > 0.6 else (0, 165, 255)
        cv2.putText(frame, f"{last_results['class'].upper()} ({last_results['conf']*100:.1f}%)", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    # This sends the cached prediction results to the frontend
    return jsonify(last_results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)