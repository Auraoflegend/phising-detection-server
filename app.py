from flask import Flask, request, jsonify
import joblib
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/file/d/13VyiL10Ge6Csk2KoU5vx0sIVfpcKX_BO/view?usp=drive_link"
MODEL_PATH = "phishing_ml_model.pkl"

# Step 1: Download model from Google Drive if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model downloaded.")
        else:
            raise Exception(f"❌ Failed to download model, status code: {response.status_code}")

# Step 2: Load the model
download_model()
model = joblib.load(MODEL_PATH)

# Step 3: Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])
        if not features or len(features) != 41:
            return jsonify({"error": "Invalid input. Must contain 41 features."}), 400

        prediction = model.predict([features])[0]
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 4: Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)