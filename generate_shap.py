"""
ProjectIQ: SHAP Explainability Engine & Visualizations
Milestone: Phase 2 - Explainability Layer & Demonstration Assets
"""

import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

print("=" * 60)
print("Loading model and dataset for SHAP explainability analysis...")

# 1. Load pipeline and data
pipeline = joblib.load("projectiq_model.joblib")
preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["classifier"]

# Load parquet sample for background explainability
df_raw = pd.read_parquet("kickstarter_dataset.parquet")
valid_states = ["successful", "failed"]
df = df_raw[df_raw["state"].isin(valid_states)].copy()
df["target"] = (df["state"] == "successful").astype(int)

# Quick parse for feature consistency
df["launched_at"] = pd.to_datetime(df["launched_at"])
df["created_at"] = pd.to_datetime(df["created_at"])
df["deadline"] = pd.to_datetime(df["deadline"])
df = df[df["launched_at"] >= pd.Timestamp("2009-01-01")].copy()
df = df[df["deadline"] > df["launched_at"]].copy()

rate = df["static_usd_rate"].fillna(1.0)
df["log_goal_usd"] = np.log1p(df["goal"] * rate)
df["campaign_duration_days"] = (df["deadline"] - df["launched_at"]).dt.total_seconds() / (24 * 3600)
df["prep_duration_days"] = (df["launched_at"] - df["created_at"]).dt.total_seconds() / (24 * 3600)
df["prep_duration_days"] = df["prep_duration_days"].clip(lower=0)

df["launch_month"] = df["launched_at"].dt.month.astype(str)
df["launch_day_of_week"] = df["launched_at"].dt.day_name()
df["launch_hour"] = df["launched_at"].dt.hour
df["name_len"] = df["name"].fillna("").astype(str).str.len()
df["name_word_count"] = df["name"].fillna("").astype(str).apply(lambda x: len(x.split()))
df["blurb_len"] = df["blurb"].fillna("").astype(str).str.len()
df["blurb_word_count"] = df["blurb"].fillna("").astype(str).apply(lambda x: len(x.split()))
df["has_video"] = df["video"].notnull().astype(int)
df["staff_pick"] = df["staff_pick"].astype(int)
df["prelaunch_activated"] = df["prelaunch_activated"].astype(int)

import re
def parse_category(val):
    if pd.isna(val): return "Other"
    if isinstance(val, dict): return val.get("parent_name", val.get("name", "Other"))
    if isinstance(val, str):
        m = re.search(r'"parent_name"\s*:\s*"([^"]+)"', val)
        if m: return m.group(1)
        m_name = re.search(r'"name"\s*:\s*"([^"]+)"', val)
        if m_name: return m_name.group(1)
        return val.split("/")[0].strip()
    return "Other"

df["category_clean"] = df["category"].apply(parse_category)
top_cats = df["category_clean"].value_counts().nlargest(15).index
df["category_clean"] = df["category_clean"].apply(lambda x: x if x in top_cats else "Other")

top_countries = df["country"].value_counts().nlargest(10).index
df["country_clean"] = df["country"].apply(lambda x: x if x in top_countries else "Other")

features = [
    "log_goal_usd", "campaign_duration_days", "prep_duration_days", "launch_hour",
    "name_len", "name_word_count", "blurb_len", "blurb_word_count", "has_video",
    "staff_pick", "prelaunch_activated", "category_clean", "country_clean",
    "launch_month", "launch_day_of_week"
]

# Sample 5,000 instances for efficient and exact SHAP calculations
sample_df = df[features].sample(n=5000, random_state=42)

print("Transforming feature matrix using fitted preprocessor...")
X_transformed = preprocessor.transform(sample_df)

# Retrieve transformed feature names
cat_encoder = preprocessor.named_transformers_["cat"]
cat_names = cat_encoder.get_feature_names_out(["category_clean", "country_clean", "launch_month", "launch_day_of_week"])
num_names = [
    "log_goal_usd", "campaign_duration_days", "prep_duration_days", "launch_hour",
    "name_len", "name_word_count", "blurb_len", "blurb_word_count", "has_video",
    "staff_pick", "prelaunch_activated"
]
all_feature_names = list(num_names) + list(cat_names)

print("Calculating SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)

# Generate SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_transformed, feature_names=all_feature_names, show=False, max_display=15)
plt.title("ProjectIQ: Top 15 Feature Impacts on Project Success (SHAP)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# Compute mean absolute SHAP for ranking
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Mean Absolute SHAP": mean_abs_shap
}).sort_values(by="Mean Absolute SHAP", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("TOP 10 MOST ACTIONABLE RISK/SUCCESS DRIVERS (SHAP):")
print(importance_df.head(10).to_string(index=False))
print("\nPlot saved successfully as 'shap_summary.png'!")
print("=" * 60)
