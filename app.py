from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Google Drive model ID
MODEL_ID = "1Mi240WxfjMNWJQ6SoWcbWG97pyCDI1xE"
MODEL_PATH = "phishing_ml_model.pkl"

# Download model using gdown
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

# Load the model safely
try:
    download_model()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available."}), 500

    data = request.get_json()
    features = data.get("features", [])

    if not isinstance(features, list) or len(features) != 41:
        return jsonify({"error": "Invalid input. Must contain 41 features."}), 400

    try:
        prediction = model.predict([features])[0]
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
