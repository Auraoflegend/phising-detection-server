from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# === Your model settings ===
MODEL_ID = "1yVIJXXy5fSNme64nBxJ8MR9A7wotkm8a"
MODEL_PATH = "phishing_ml_model.pkl"

# === Download model if not exists ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive using gdown...")
        try:
            import gdown
        except ImportError:
            os.system("pip install gdown")
            import gdown
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
        print("✅ Model downloaded.")

# === Load model ===
try:
    download_model()
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded and ready.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# === Predict endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available."}), 500

    data = request.get_json()
    features = data.get("features", [])

    if not features or len(features) != 41:
        return jsonify({"error": "Invalid input. Must contain 41 features."}), 400

    try:
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0][1]
        label = "phishing" if prediction == 1 else "legitimate"
        return jsonify({
            "result": label,
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Health check ===
@app.route("/")
def index():
    return "🔒 Phishing Detector API is running."

# === Run the app ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
