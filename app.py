import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
with open("phishing_ml_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting all 41 features in the right order
    if not data or not isinstance(data.get("features"), list):
        return jsonify({"error": "Expected a JSON with 'features': [ ... ]"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "phishing" if prediction == 1 else "legitimate"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

