# utils/helper.py
# Helper functions for charts and feature importance.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def get_grade(score):
    """Convert numeric score to letter grade."""
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"

def score_color(score):
    """Return a hex color based on exam score."""
    if score >= 70: return "#2ecc71"   # green
    if score >= 50: return "#f39c12"   # orange
    return "#e74c3c"                   # red

def stress_color(label):
    """Return a hex color based on stress label."""
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(label, "#888")

def plot_feature_importance(model, features):
    """Bar chart of feature importances from the Random Forest."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [features[i] for i in indices]
    sorted_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1],
                   color="#7C5CBF", edgecolor="none")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("What affects your exam score the most?", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_radar(values, labels):
    """Simple bar chart styled to look like a habit summary."""
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#3498db", "#9b59b6", "#e74c3c", "#2ecc71", "#f39c12", "#1abc9c", "#e67e22"]
    ax.bar(labels, values, color=colors[:len(labels)], edgecolor="none")
    ax.set_title("Your daily habits at a glance", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9, rotation=15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig