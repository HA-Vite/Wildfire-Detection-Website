# we designed the 'extensions.py' file to plug-in the new features into the backend without editing 
# the Flask server 'app.py' code so it remains clear to you and doesn't get too long and complicated 

# It provides the Live Map catalog , the Grad-CAM Explainable-AI endpoint
# and the per-browser scan history with statistics

import base64
import datetime
import io
import json
import os
import re
import threading
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import jsonify, request

# Lock that keeps the history file consistent under parallel requests
_HISTORY_LOCK = threading.Lock()

# Cookie that identifies a browser so each user only sees their own scans
CLIENT_COOKIE = "wf_client"


def register_extensions(app, model, run_ai_inference, quick_images_dir):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(base_dir, "scan_history.json")

    # - History Storage Section : one JSON file holding the client_id and the records

    def _load_all():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _records_for(client_id):
        return _load_all().get(client_id, [])

    def _append_history(client_id, entry):
        with _HISTORY_LOCK:
            data = _load_all()
            records = data.get(client_id, [])
            records.append(entry)
            # Keep only the latest 300 records per browser to protect disk space
            data[client_id] = records[-300:]
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # - Client_ID Section : reads the browser cookie , the after_request hook sets the cookie
    def _client_id():
        return request.cookies.get(CLIENT_COOKIE)

    # observes the prediction endpoints and records each scan under the requesting browser
    @app.after_request
    def _log_scan_activity(response):
        try:
            client_id = _client_id()
            if not client_id:
                client_id = uuid.uuid4().hex
                response.set_cookie(
                    CLIENT_COOKIE, client_id,
                    max_age=60 * 60 * 24 * 365, samesite="Lax"
                )

            watched = ("/predict", "/predict-quick", "/predict-explain")
            if request.path in watched and response.status_code == 200:
                payload = response.get_json(silent=True) or {}
                if "prediction" in payload:
                    if request.path == "/predict-quick":
                        body = request.get_json(silent=True) or {}
                        image_name = str(body.get("image_name", "unknown"))
                    else:
                        file_obj = request.files.get("image")
                        image_name = getattr(file_obj, "filename", None) or "uploaded_image"

                    _append_history(client_id, {
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": request.path,
                        "image": image_name,
                        "prediction": payload.get("prediction"),
                        "confidence": payload.get("confidence"),
                    })
        except Exception as log_err:
            # Logging must never break a working response
            print(f"[HISTORY LOG SKIPPED] -> {log_err}")
        return response

    # - Live Map Section : quick-scan images are named "longitude,latitude.jpg" ,
    # parse those names into real geographic marker positions to appear on the map
    @app.route("/quick-list", methods=["GET"])
    def quick_list():
        try:
            items = []
            coord_pattern = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")
            for fname in sorted(os.listdir(quick_images_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = os.path.splitext(fname)[0]
                match = coord_pattern.match(stem)
                if match:
                    items.append({
                        "name": fname,
                        "lon": float(match.group(1)),
                        "lat": float(match.group(2)),
                        "url": f"/quickimages/{fname}",
                    })
            return jsonify({"images": items, "count": len(items)}), 200
        except Exception as e:
            print(f"[EXCEPTION LOG - QUICK LIST] -> {str(e)}")
            return jsonify({"error": "Could not build the map catalog."}), 500

    # - Grad-CAM AI Insight Section : shows where the model looked by weighting the last MobileNetV2
    # convolutional layer with the gradients of the predicted class

    _grad_cache = {}

    def _get_grad_components():
        if "components" in _grad_cache:
            return _grad_cache["components"]

        # Locate the nested MobileNetV2 base inside the Sequential model
        base = None
        base_index = -1
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.Model):
                base = layer
                base_index = i
                break
        if base is None:
            raise RuntimeError("No nested convolutional base found in model")

        last_conv = base.get_layer("Conv_1")  # Final conv layer of MobileNetV2
        grad_model = tf.keras.models.Model(
            base.input, [last_conv.output, base.output]
        )
        head_layers = model.layers[base_index + 1:]  # GAP + Dropout + Dense

        _grad_cache["components"] = (grad_model, head_layers)
        return _grad_cache["components"]

    def _compute_gradcam(image_array):
        grad_model, head_layers = _get_grad_components()

        with tf.GradientTape() as tape:
            conv_output, features = grad_model(image_array)
            x = features
            for layer in head_layers:
                x = layer(x, training=False)
            predictions = x
            predicted_index = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_index]

        grads = tape.gradient(class_score, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    def _render_heatmap_overlay(pil_image, heatmap):
        # Upscale the small heatmap smoothly to the original image size
        heat_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), "L")
        heat_img = heat_img.resize(pil_image.size, Image.BILINEAR)
        heat = np.array(heat_img).astype(np.float32) / 255.0

        # Hot colormap : black -> red -> orange -> yellow
        r = np.clip(heat * 3.0, 0, 1) * 255
        g = np.clip(heat * 3.0 - 1.0, 0, 1) * 255
        b = np.clip(heat * 3.0 - 2.0, 0, 1) * 255
        color_layer = np.stack([r, g, b], axis=-1).astype(np.float32)

        base_arr = np.array(pil_image.convert("RGB")).astype(np.float32)
        alpha = (0.70 * (heat ** 1.5))[..., None]  # Low attention stays transparent , hot zones glow
        blended = (base_arr * (1 - alpha) + color_layer * alpha).astype(np.uint8)
        return Image.fromarray(blended)

    def _pil_to_data_url(pil_image):
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @app.route("/predict-explain", methods=["POST"])
    def predict_explain():
        # Same payload validation philosophy as the core endpoints
        if "image" not in request.files:
            return jsonify({"error": "No file packet payload received"}), 400

        file = request.files["image"]
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            return jsonify({"error": "Unsupported file media type configuration"}), 415

        try:
            raw_bytes = file.read()
            image = Image.open(io.BytesIO(raw_bytes))

            # Standard prediction through the untouched core inference pipeline
            predicted_class, confidence = run_ai_inference(image)

            # Grad-CAM preprocessing mirrors training : rescale to [0,1] like the AI Model was trained
            resized = image.convert("RGB").resize((224, 224))
            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            try:
                heatmap = _compute_gradcam(arr)
                display_img = image.convert("RGB")
                # Cap the display size to keep the response payload light
                display_img.thumbnail((640, 640))
                overlay = _render_heatmap_overlay(display_img, heatmap)
                heatmap_data_url = _pil_to_data_url(overlay)
                explain_available = True
            except Exception as cam_err:
                print(f"[GRAD-CAM UNAVAILABLE] -> {cam_err}")
                heatmap_data_url = None
                explain_available = False

            return jsonify({
                "prediction": predicted_class,
                "confidence": confidence,
                "explain_available": explain_available,
                "heatmap": heatmap_data_url,
            }), 200

        except Exception as e:
            print(f"[EXCEPTION LOG - EXPLAIN] -> {str(e)}")
            return jsonify({"error": "Internal error occurred during explainability loop."}), 500

    # - History Read Section : returns this browser's own records and their statistics
    @app.route("/api/history", methods=["GET"])
    def get_history():
        try:
            records = _records_for(_client_id())
            total = len(records)
            fires = sum(1 for r in records if str(r.get("prediction", "")).lower() == "wildfire")
            safe = total - fires
            confidences = [float(r.get("confidence", 0)) for r in records if r.get("confidence") is not None]
            avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0

            return jsonify({
                "records": list(reversed(records)),  # Newest first
                "stats": {
                    "total": total,
                    "wildfire": fires,
                    "nowildfire": safe,
                    "avg_confidence": avg_conf,
                    "last_scan": records[-1]["time"] if records else None,
                },
            }), 200
        except Exception as e:
            print(f"[EXCEPTION LOG - HISTORY] -> {str(e)}")
            return jsonify({"error": "Internal error occurred during history lookup."}), 500

    # - History Clear Section : wipes only the requesting browser's own records
    @app.route("/api/history", methods=["DELETE"])
    def clear_history():
        try:
            client_id = _client_id()
            with _HISTORY_LOCK:
                data = _load_all()
                if client_id in data:
                    data.pop(client_id, None)
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            return jsonify({"message": "History cleared successfully"}), 200
        except Exception as e:
            print(f"[EXCEPTION LOG - HISTORY CLEAR] -> {str(e)}")
            return jsonify({"error": "Internal error occurred while clearing history."}), 500

    print("Feature extensions registered : /quick-list , /predict-explain , /api/history")
