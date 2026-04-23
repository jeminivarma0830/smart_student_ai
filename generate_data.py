# data/generate_data.py
# This script generates a synthetic student dataset for training the AI model.

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

study_hours     = np.random.uniform(1, 10, n)
sleep_hours     = np.random.uniform(4, 9, n)
screen_time     = np.random.uniform(1, 8, n)
exercise_mins   = np.random.uniform(0, 90, n)
social_media    = np.random.uniform(0, 6, n)
attendance_pct  = np.random.uniform(50, 100, n)
diet_quality    = np.random.randint(1, 6, n)   # 1=poor, 5=excellent

# Formula to calculate exam score (based on realistic weights)
exam_score = (
    study_hours    * 5.0 +
    sleep_hours    * 2.5 +
    attendance_pct * 0.3 +
    exercise_mins  * 0.1 +
    diet_quality   * 1.5 -
    screen_time    * 2.0 -
    social_media   * 1.5 +
    np.random.normal(0, 5, n)   # add some noise
)

# Clip score to 0–100 range
exam_score = np.clip(exam_score, 0, 100).round(1)

# Calculate stress level: Low / Medium / High
stress_raw = (
    screen_time   * 2.0 +
    social_media  * 1.5 -
    sleep_hours   * 2.0 -
    exercise_mins * 0.05 -
    study_hours   * 0.5 +
    np.random.normal(0, 2, n)
)

stress_level = pd.cut(
    stress_raw,
    bins=[-999, 5, 10, 999],
    labels=["Low", "Medium", "High"]
)

df = pd.DataFrame({
    "study_hours":    study_hours.round(1),
    "sleep_hours":    sleep_hours.round(1),
    "screen_time":    screen_time.round(1),
    "exercise_mins":  exercise_mins.round(0),
    "social_media":   social_media.round(1),
    "attendance_pct": attendance_pct.round(1),
    "diet_quality":   diet_quality,
    "exam_score":     exam_score,
    "stress_level":   stress_level
})

df.to_csv("data/student_data.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows saved to data/student_data.csv")