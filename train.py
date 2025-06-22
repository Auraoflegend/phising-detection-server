import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Assume 'Type' is the target column
y = df["Type"]

# Drop the label to get all features
X = df.drop(columns=["Type"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy scores
print("✅ Training accuracy:", model.score(X_train, y_train))
print("✅ Test accuracy:", model.score(X_test, y_test))

# Save the model
with open("phishing_ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as phishing_ml_model.pkl")
