import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ✅ Step 1: Load the dataset (already cleaned, no NaNs assumed)
df = pd.read_csv("dataset.csv")

# ✅ Step 2: Define features and label
X = df.drop(columns=["Type"])
y = df["Type"]

# ✅ Step 3: Train-test split (stratify ensures class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Step 4: Train optimized lightweight model
model = RandomForestClassifier(
    n_estimators=30,         # Fewer trees
    max_depth=10,            # Limit depth to reduce memory
    max_features="sqrt",     # Use subset of features per tree
    n_jobs=1,                # Avoid parallel threads (important for Render)
    random_state=42
)
model.fit(X_train, y_train)

# ✅ Step 5: Print accuracy for verification
print("✅ Training accuracy:", model.score(X_train, y_train))
print("✅ Test accuracy:", model.score(X_test, y_test))

# ✅ Step 6: Save with compression to reduce file size
joblib.dump(model, "phishing_ml_model.pkl", compress=3)
print("✅ Model saved as phishing_ml_model.pkl")
