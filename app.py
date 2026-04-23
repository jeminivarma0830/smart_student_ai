# app.py
# SmartStudentAI — Main Streamlit Web App
# Run with: streamlit run app.py

import streamlit as st
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils.helper import (
    get_grade, score_color, stress_color,
    plot_feature_importance, plot_radar
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartStudentAI",
    page_icon="🎓",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model()
except FileNotFoundError:
    st.error("⚠️ Model not found! Run these commands first:\n"
             "```\npython data/generate_data.py\npython model/train_model.py\n```")
    st.stop()

score_model  = bundle["score_model"]
stress_model = bundle["stress_model"]
le           = bundle["label_encoder"]
features     = bundle["features"]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🎓 SmartStudentAI")
st.markdown("**Predict your exam score and stress level based on your daily habits.**")
st.markdown("---")

# ── Sidebar: Input sliders ─────────────────────────────────────────────────
st.sidebar.header("📋 Enter Your Daily Habits")

study_hours    = st.sidebar.slider("📚 Study hours per day",       0.0, 12.0, 4.0, 0.5)
sleep_hours    = st.sidebar.slider("😴 Sleep hours per night",     4.0,  9.0, 7.0, 0.5)
screen_time    = st.sidebar.slider("📱 Total screen time (hrs)",   0.0,  8.0, 3.0, 0.5)
exercise_mins  = st.sidebar.slider("🏃 Exercise (minutes/day)",    0.0, 90.0,30.0, 5.0)
social_media   = st.sidebar.slider("💬 Social media (hrs/day)",    0.0,  6.0, 2.0, 0.5)
attendance_pct = st.sidebar.slider("🏫 Class attendance (%)",     50.0,100.0,80.0, 1.0)
diet_quality   = st.sidebar.selectbox(
    "🥗 Diet quality",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {1:"1 - Very Poor", 2:"2 - Poor", 3:"3 - Average",
                            4:"4 - Good", 5:"5 - Excellent"}[x],
    index=2
)

predict_btn = st.sidebar.button("🔮 Predict Now!", use_container_width=True)

# ── Main area ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

if predict_btn:
    # Build input array
    input_data = np.array([[study_hours, sleep_hours, screen_time,
                            exercise_mins, social_media, attendance_pct, diet_quality]])

    # Predictions
    predicted_score  = score_model.predict(input_data)[0]
    predicted_score  = round(float(np.clip(predicted_score, 0, 100)), 1)
    stress_idx       = stress_model.predict(input_data)[0]
    predicted_stress = le.inverse_transform([stress_idx])[0]
    grade            = get_grade(predicted_score)

    # ── Result cards ──────────────────────────────────────────────────────
    with col1:
        color = score_color(predicted_score)
        st.markdown(f"""
        <div style="background:{color}22;border-left:5px solid {color};
             padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:{color};margin:0">{predicted_score}/100</h2>
            <p style="margin:4px 0 0;font-size:13px">Predicted Exam Score</p>
            <h3 style="color:{color};margin:6px 0 0">Grade: {grade}</h3>
        </div>""", unsafe_allow_html=True)

    with col2:
        scolor = stress_color(predicted_stress)
        st.markdown(f"""
        <div style="background:{scolor}22;border-left:5px solid {scolor};
             padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:{scolor};margin:0">{predicted_stress}</h2>
            <p style="margin:4px 0 0;font-size:13px">Predicted Stress Level</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background:#3498db22;border-left:5px solid #3498db;
             padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#3498db;margin:0">{attendance_pct:.0f}%</h2>
            <p style="margin:4px 0 0;font-size:13px">Your Attendance</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 What drives your score?")
        fig1 = plot_feature_importance(score_model, features)
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("📈 Your habits overview")
        habit_vals   = [study_hours, sleep_hours, screen_time,
                        exercise_mins/10, social_media, attendance_pct/10, diet_quality]
        habit_labels = ["Study", "Sleep", "Screen", "Exercise/10",
                        "SocialMedia", "Attend/10", "Diet"]
        fig2 = plot_radar(habit_vals, habit_labels)
        st.pyplot(fig2)

    # ── Personalised tips ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Personalised Tips")

    tips = []
    if study_hours < 4:
        tips.append("📚 Try to study at least 4–5 hours daily for better results.")
    if sleep_hours < 7:
        tips.append("😴 You're under-sleeping! Aim for 7–8 hours — it boosts memory.")
    if screen_time > 5:
        tips.append("📵 Reduce screen time — anything above 5 hrs hurts concentration.")
    if exercise_mins < 20:
        tips.append("🏃 Add at least 20–30 minutes of exercise daily to reduce stress.")
    if social_media > 3:
        tips.append("💬 Social media over 3 hrs/day is hurting your focus. Try app timers.")
    if attendance_pct < 75:
        tips.append("🏫 Attendance below 75% can seriously hurt your grades — show up!")
    if diet_quality <= 2:
        tips.append("🥗 Poor diet affects energy and focus. Include fruits and proteins.")
    if not tips:
        tips.append("🌟 Amazing habits! Keep it up — you're on track for a great semester.")

    for tip in tips:
        st.info(tip)

else:
    # Welcome state (before clicking Predict)
    with col1:
        st.info("👈 Adjust the sliders in the sidebar and click **Predict Now!**")
    with col2:
        st.markdown("### What this app predicts:")
        st.markdown("- 📊 **Exam Score** (0–100)\n- 😰 **Stress Level** (Low/Medium/High)\n- 💡 **Personalised study tips**")
    with col3:
        st.markdown("### How it works:")
        st.markdown("1. Enter your daily habits\n2. AI model analyses patterns\n3. Get instant predictions + tips")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🤖 Built with Python, Scikit-learn & Streamlit | BScIT AI Project")