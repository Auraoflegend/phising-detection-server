import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load dataset
df = pd.read_csv("dataset.csv")

# Step 2: Extract features and target
X = df.drop(columns=["Type"])
y = df["Type"]

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"✅ Training accuracy: {train_acc:.4f}")
print(f"✅ Test accuracy: {test_acc:.4f}")

# Step 6: Save model using joblib for cross-version compatibility
joblib.dump(model, "phishing_ml_model.pkl", compress=3, protocol=4)
print("✅ Model saved as phishing_ml_model.pkl using joblib")
