from flask import Flask, request, jsonify
import joblib
import os
import requests

app = Flask(__name__)

# 📦 Google Drive model file (Direct Download Link)
MODEL_URL = "https://drive.google.com/file/d/10jqPKx0pVaougdgd4m4g9bviiCOZxLtE"
MODEL_PATH = "phishing_ml_model.pkl"

# 🔽 Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model downloaded.")
        else:
            raise Exception(f"❌ Failed to download model: {response.status_code}")

# 🧠 Load the model
try:
    download_model()
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

# 🔍 Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    features = data.get("features")

    if not features or len(features) != 41:
        return jsonify({"error": "Invalid input: must provide 41 features."}), 400

    try:
        prediction = model.predict([features])[0]
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 Run server (for local testing)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
