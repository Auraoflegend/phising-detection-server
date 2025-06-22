import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

# === Step 1: Load dataset ===
df = pd.read_csv("Dataset.csv")

# === Step 2: Split features and label ===
X = df.drop(columns=["Type"])
y = df["Type"]

# === Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 4: Initialize RandomForest with tuned parameters ===
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,               # Limit depth to reduce overfitting
    min_samples_split=4,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# === Step 5: Train the model ===
model.fit(X_train, y_train)

# === Step 6: Evaluate the model ===
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"✅ Training accuracy: {train_acc:.4f}")
print(f"✅ Test accuracy: {test_acc:.4f}")

# === Step 7: Cross-validation score ===
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n📈 Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# === Step 8: Save the model ===
joblib.dump(model, "phishing_ml_model.pkl")
print("✅ Model saved as phishing_ml_model.pkl")
