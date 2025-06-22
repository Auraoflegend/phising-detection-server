from flask import Flask, request, jsonify
import joblib
import os
import requests

app = Flask(__name__)

# ✅ Use new direct link from Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=114CSoYogPl9iRTnEm_6NDSOFpA5ROwi8"
MODEL_PATH = "phishing_ml_model.pkl"

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

# Load the model
try:
    download_model()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
