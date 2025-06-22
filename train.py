import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # ✅ use joblib here

# Load dataset
df = pd.read_csv("dataset.csv")

# Separate features and labels
X = df.drop(columns=["Type"])
y = df["Type"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print("✅ Train Accuracy:", model.score(X_train, y_train))
print("✅ Test Accuracy:", model.score(X_test, y_test))

# ✅ Save model with joblib
joblib.dump(model, "phishing_ml_model.pkl")
print("✅ Model saved as phishing_ml_model.pkl")
