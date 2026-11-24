"""
Step 2: Train and compare three basic ML models.

Models: Logistic Regression, Random Forest, XGBoost.
The best one (highest F1 score) is saved in the models/ folder.

Run:  python src/train_model.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load the data
df = pd.read_csv("data/symptoms_diseases.csv")

X = df.drop(columns=["disease"])   # symptoms (0 or 1)
y = df["disease"]                  # disease name

# 2. Turn disease names into numbers (models need numbers)
label_encoder = LabelEncoder()
y_numbers = label_encoder.fit_transform(y)

# 3. Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numbers, test_size=0.2, random_state=42, stratify=y_numbers
)

# 4. The models we want to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=42),
}

# 5. Train each model and check its score
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    results.append({"model": name, "accuracy": round(accuracy, 4), "f1_score": round(f1, 4)})
    trained_models[name] = model
    print(f"{name:20s} accuracy = {accuracy:.3f}   f1 = {f1:.3f}")

results_df = pd.DataFrame(results)

# 6. Pick the best model by F1 score
best_name = results_df.sort_values("f1_score", ascending=False).iloc[0]["model"]
best_model = trained_models[best_name]
print("\nBest model:", best_name)

# 7. Detailed report for the best model
best_predictions = best_model.predict(X_test)
print("\n", classification_report(
    y_test, best_predictions, target_names=label_encoder.classes_
))

# 8. Save everything the app needs
joblib.dump(best_model, "models/model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(list(X.columns), "models/symptoms.pkl")
results_df.to_csv("models/model_results.csv", index=False)
print("Saved model files in the models/ folder.")
