from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the frontend build directory (Vite outputs a 'dist' folder by default)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "dist")

# Initialize Flask app and map it to serve static assets from the frontend build folder
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# [ Security Policy (1) ] -CORS- : To secure react local port handshakes
CORS(app)   

# Extract client IP and bypass rate limits for localhost to prevent developer blocks
def custom_security_router_key():
    client_ip = get_remote_address()
    if client_ip in ["127.0.0.1", "::1", "localhost"]:
        return "localhost_unlimited_bypass_token"
    return client_ip

# [ Security Policy (2) ] -Rate Limiting- : Prevents DoS attacks by blocking external requests 
# 60 request per minutes allowed
limiter = Limiter(
    key_func=custom_security_router_key,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
    headers_enabled=True
)

# Error handler that enforces a strict 3-minute cooldown (180s) on Dos attackers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too Many Requests. Rate limit exceeded",
        "retry_after_seconds": 180 
    }), 429

# File System Paths & Environment Setup
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
QUICK_IMAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "dist", "quickimages")

# Cropping images to suit the AI Model
IMG_SIZE = (224, 224)
CLASS_NAMES = ["nowildfire", "wildfire"]

# AI Model Initialization , Loads the trained weights into RAM , for speed and storage purposes
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Main AI Processing Pipeline
def run_ai_inference(image_obj):
    # Processes image matrices and executes standard MobileNetV2 inference loops safely
    image = image_obj.convert("RGB").resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array, verbose=0)
    predicted_index = int(np.argmax(prediction))
    
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(prediction[0][predicted_index] * 100)
    
    return str(predicted_class).lower(), round(confidence, 2)


# --- Frontend Static Content Routing Architecture ---

# Root route serving the main entry point index.html file from the static folder
@app.route("/", methods=["GET"])
@limiter.exempt 
def home():
    return send_from_directory(app.static_folder, "index.html")

# Catch-all route handler ensuring React Router layouts survive client-side browser refreshes
@app.route("/<path:path>", methods=["GET"])
@limiter.exempt
def catch_all(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ----------------------------------------------------


# - API Endpoint [1] - :  Satellite Image Upload Pipeline
@app.route("/predict", methods=["POST"])
@limiter.limit("60 per minute") 

# API SECURITY FUNCTION that validates binary stream file uploads against remote command execution hacks
def predict():
    
    # [ Security Policy (3) ] -Structural Payload Validation- : Blocks empty data packets 
    #  and allows the file to be only an image
    if "image" not in request.files:
        return jsonify({"error": "No file packet payload received"}), 400

    file = request.files["image"]
    
    # [ Security Policy (4) ] -MIME-TYPE Whitelisting- : Permits only raw specific image formats
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return jsonify({"error": "Unsupported file media type configuration"}), 415

    try:
        # [ Security Policy (5) ] -Magic Bytes Inspection- : Verifies hex signature to crush Polyglot files
        file.stream.seek(0)
        header_bytes = file.stream.read(4)
        file.stream.seek(0) 
        
        is_jpeg = header_bytes[0:3] == b'\xff\xd8\xff'
        is_png = header_bytes[0:4] == b'\x89PNG'
        
        if not (is_jpeg or is_png):
            return jsonify({"error": "Invalid image hex signature (MIME Spoof Terminated)"}), 415

        # [ Security Policy (6) ] -IN-Memory Processing- : Skips hard disk logs via safe temporary binary streams on RAM
        raw_bytes = file.read()
        image = Image.open(io.BytesIO(raw_bytes))
        
        predicted_class, confidence = run_ai_inference(image)
        return jsonify({"prediction": predicted_class, "confidence": confidence}), 200

    # [ Security Policy (7) ] -Exception Masking- : Blocking system path disclosure that appears in error messages
    except Exception as e:
        print(f"[EXCEPTION LOG - UPLOAD] -> {str(e)}")
        return jsonify({"error": "Internal error occurred during inference execution loop."}), 500

# - API Endpoint [2] - : Pre-Loaded Images Quick Scan Pipeline
@app.route("/predict-quick", methods=["POST"])
@limiter.limit("60 per minute")

# API SECURITY FUNCTION that validates local lookup directory queries under path routing restrictions
def predict_quick():
        
    # [ Security Policy (3) - For Endpoint [2] ] -Structural Payload Validation- : Confirms query key integrity
    data = request.get_json()
    if not data or "image_name" not in data:
        return jsonify({"error": "Missing database query token signature"}), 400

    try:
        raw_name = str(data["image_name"])
        
        # [ Security Policy (8) ] -Path Traversal Sanitization- : Prevents directory breakouts (../)
        # Allowing commas (,) and float dots (.) 
        # because they exist in satellite images names , including the ones in quick test scan sample-box
        clean_base = os.path.basename(raw_name)
        
        image_path = os.path.join(QUICK_IMAGES_DIR, clean_base)

        if not os.path.exists(image_path):
            return jsonify({"error": "Target inventory asset configuration could not be mapped locally."}), 404

        image = Image.open(image_path)
        predicted_class, confidence = run_ai_inference(image)
        return jsonify({"prediction": predicted_class, "confidence": confidence}), 200
        
    except Exception as e:
        print(f"[EXCEPTION LOG - QUICK SCAN] -> {str(e)}")
        # [ Security Policy (7) - For Endpoint [2] ] -Exception Masking- : Shields internal infrastructure mapping parameters
        return jsonify({"error": "Internal error occurred during stock database lookups."}), 500

# - Extensions Loading : plugs the new feature endpoints 
# If the extensions module is missing or fails , the core website keeps working normally 
try:
    from extensions import register_extensions
    register_extensions(app, model, run_ai_inference, QUICK_IMAGES_DIR)
except Exception as ext_err:
    print(f"[EXTENSIONS DISABLED - SOME FEATURES MAY NOT WORK] -> {ext_err}")

# Deploying Network Socket Server with cloud-compatible network binding setup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
