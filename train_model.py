# model/train_model.py
# Trains the Random Forest models and saves them to disk.

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/student_data.csv")
print(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

FEATURES = ["study_hours", "sleep_hours", "screen_time",
            "exercise_mins", "social_media", "attendance_pct", "diet_quality"]

X = df[FEATURES]

# ── Model 1: Predict exam_score (Regression) ───────────────────────────────
y_score = df["exam_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)

score_model = RandomForestRegressor(n_estimators=100, random_state=42)
score_model.fit(X_train, y_train)

preds = score_model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
print(f"✅ Exam Score Model — Mean Absolute Error: {mae:.2f} marks")

# ── Model 2: Predict stress_level (Classification) ────────────────────────
le = LabelEncoder()
y_stress = le.fit_transform(df["stress_level"])   # Low=0, Medium=1, High=2

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_stress, test_size=0.2, random_state=42)

stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
stress_model.fit(X_train2, y_train2)

acc = accuracy_score(y_test2, stress_model.predict(X_test2))
print(f"✅ Stress Level Model — Accuracy: {acc*100:.1f}%")

# ── Save everything into one .pkl file ────────────────────────────────────
os.makedirs("model", exist_ok=True)
bundle = {
    "score_model":  score_model,
    "stress_model": stress_model,
    "label_encoder": le,
    "features":     FEATURES
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("💾 Models saved to model/model.pkl")