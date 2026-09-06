import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import uuid
from datetime import datetime
from threading import Lock
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Advanced environment tweaking to reduce memory overhead and block memory leaks
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.disable_eager_execution()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "dist")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, supports_credentials=True)

# Limiter Engine configuration to avoid flooding pressure
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per day", "60 per hour"],
    storage_uri="memory://"
)

# Core models and catalogs configurations paths
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
QUICK_IMAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "dist", "quickimages")
history_path = os.path.join(BASE_DIR, "scan_history.json")

_HISTORY_LOCK = Lock()
IMG_SIZE = (224, 224)
CLASS_NAMES = ["nowildfire", "wildfire"]

print("Loading model inside core inference pipeline...")
# Allocating a clean isolated graphic frame session to stabilize memory utilization
global_graph = tf.compat.v1.get_default_graph()
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def run_ai_inference(image_obj):
    global global_graph
    image = image_obj.convert("RGB").resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Executing calculation array within the boundaries of the main memory graph
    with global_graph.as_default():
        prediction = model.predict(image_array, verbose=0)
        
    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(prediction[predicted_index] * 100)
    
    # Destroys and flushes session variables immediately to prevent OOM
    tf.keras.backend.clear_session()
    
    return str(predicted_class).lower(), round(confidence, 2)

def _client_id():
    token = request.cookies.get("client_session_token")
    if not token:
        token = str(uuid.uuid4())
    return token

def _load_all():
    if not os.path.exists(history_path):
        return {}
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _records_for(cid):
    return _load_all().get(cid, [])

def _append_record(cid, record):
    with _HISTORY_LOCK:
        data = _load_all()
        if cid not in data:
            data[cid] = []
        data[cid].append(record)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file packet payload received"}), 400
        
    file = request.files["image"]
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return jsonify({"error": "Unsupported file media type configuration"}), 415
        
    try:
        raw_bytes = file.read()
        image = Image.open(io.BytesIO(raw_bytes))
        
        predicted_class, confidence = run_ai_inference(image)
        client_token = _client_id()
        
        new_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "/predict",
            "image": file.filename or "uploaded_user_image.jpg",
            "prediction": predicted_class,
            "confidence": confidence
        }
        _append_record(client_token, new_record)
        
        resp = make_response(jsonify({"prediction": predicted_class, "confidence": confidence}), 200)
        resp.set_cookie("client_session_token", client_token, max_age=31536000, httponly=True, samesite="Lax")
        return resp
    except Exception as e:
        print(f"[EXCEPTION LOG] -> {str(e)}")
        return jsonify({"error": "Internal error occurred during inference pipeline loop."}), 500

@app.route("/predict-quick", methods=["POST"])
def predict_quick():
    data = request.get_json() or {}
    image_name = data.get("image_name")
    if not image_name:
        return jsonify({"error": "Missing image reference name payload"}), 400
        
    target_path = os.path.join(QUICK_IMAGES_DIR, image_name)
    if not os.path.exists(target_path):
        return jsonify({"error": "Requested image resource not found on node spatial catalog"}), 404
        
    try:
        image = Image.open(target_path)
        predicted_class, confidence = run_ai_inference(image)
        client_token = _client_id()
        
        new_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "/predict-quick",
            "image": image_name,
            "prediction": predicted_class,
            "confidence": confidence
        }
        _append_record(client_token, new_record)
        
        resp = make_response(jsonify({"prediction": predicted_class, "confidence": confidence}), 200)
        resp.set_cookie("client_session_token", client_token, max_age=31536000, httponly=True, samesite="Lax")
        return resp
    except Exception as e:
        print(f"[EXCEPTION LOG] -> {str(e)}")
        return jsonify({"error": "Internal node system error during catalog scanning."}), 500

# Initializing advanced layout features extensions routing mechanisms
from extensions import register_extensions
register_extensions(app, run_ai_inference, _client_id, _records_for, _append_record, _HISTORY_LOCK, history_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
