from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_ID = "1Mi240WxfjMNWJQ6SoWcbWG97pyCDI1xE"  # your latest model ID
MODEL_PATH = "phishing_ml_model.pkl"

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

# Load model safely
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
        proba = model.predict_proba([features])[0][1]  # Confidence of being phishing
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({
            "result": result,
            "confidence": round(float(proba), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
