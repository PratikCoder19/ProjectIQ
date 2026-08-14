"""
ProjectIQ: AI-Driven Project Success & Risk Decision Support System
Milestone: Phase 2 - Preprocessing, Baseline & Model Benchmarking
Prediction Point: T_0 (Launch Time - Strict Leakage Prevention)
"""

import json
import os
import re
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA
# ==========================================
PARQUET_FILE_PATH = "kickstarter_dataset.parquet"  # <-- Adjust filename if needed

print("=" * 60)
print(f"Loading dataset from: {PARQUET_FILE_PATH}")
if not os.path.exists(PARQUET_FILE_PATH):
    # Try finding any parquet file in the current directory
    parquet_files = [f for f in os.listdir(".") if f.endswith(".parquet")]
    if parquet_files:
        PARQUET_FILE_PATH = parquet_files[0]
        print(f"Found Parquet file automatically: {PARQUET_FILE_PATH}")
    else:
        raise FileNotFoundError("No .parquet file found in current directory!")

df_raw = pd.read_parquet(PARQUET_FILE_PATH)
print(f"Raw Data Loaded: {df_raw.shape[0]:,} rows | {df_raw.shape[1]} columns")

# ==========================================
# 2. TARGET FILTERING & CLEANING
# ==========================================
print("\n" + "=" * 60)
print("Filtering target classes and date epoch artifacts...")

# Filter only resolved project outcomes
valid_states = ["successful", "failed"]
df = df_raw[df_raw["state"].isin(valid_states)].copy()
df["target"] = (df["state"] == "successful").astype(int)

# Filter epoch 1970 artifacts (Kickstarter launched in 2009)
df["launched_at"] = pd.to_datetime(df["launched_at"])
df["created_at"] = pd.to_datetime(df["created_at"])
df["deadline"] = pd.to_datetime(df["deadline"])

df = df[df["launched_at"] >= pd.Timestamp("2009-01-01")].copy()
df = df[df["deadline"] > df["launched_at"]].copy()

print(f"Cleaned dataset: {df.shape[0]:,} rows")
print(f"Class Balance (Target = 1 [Successful]): {df['target'].mean() * 100:.2f}%")

# ==========================================
# 3. PRE-LAUNCH FEATURE ENGINEERING (T_0)
# ==========================================
print("\n" + "=" * 60)
print("Engineering pre-launch features (No Target Leakage)...")

# Goal in USD
rate = df["static_usd_rate"].fillna(1.0)
df["goal_usd"] = df["goal"] * rate
df["log_goal_usd"] = np.log1p(df["goal_usd"])

# Temporal features
df["campaign_duration_days"] = (df["deadline"] - df["launched_at"]).dt.total_seconds() / (24 * 3600)
df["prep_duration_days"] = (df["launched_at"] - df["created_at"]).dt.total_seconds() / (24 * 3600)
df["prep_duration_days"] = df["prep_duration_days"].clip(lower=0)

df["launch_month"] = df["launched_at"].dt.month.astype(str)
df["launch_day_of_week"] = df["launched_at"].dt.day_name()
df["launch_hour"] = df["launched_at"].dt.hour

# Content / Text signals
df["name_len"] = df["name"].fillna("").astype(str).str.len()
df["name_word_count"] = df["name"].fillna("").astype(str).apply(lambda x: len(x.split()))
df["blurb_len"] = df["blurb"].fillna("").astype(str).str.len()
df["blurb_word_count"] = df["blurb"].fillna("").astype(str).apply(lambda x: len(x.split()))
df["has_video"] = df["video"].notnull().astype(int)

# Creator / Readiness signals
df["staff_pick"] = df["staff_pick"].astype(int)
df["prelaunch_activated"] = df["prelaunch_activated"].astype(int)

# Category extraction (JSON or string parsing)
def parse_category(val):
    if pd.isna(val):
        return "Unknown"
    if isinstance(val, dict):
        return val.get("parent_name", val.get("name", "Unknown"))
    if isinstance(val, str):
        match = re.search(r'"parent_name"\s*:\s*"([^"]+)"', val)
        if match:
            return match.group(1)
        match_name = re.search(r'"name"\s*:\s*"([^"]+)"', val)
        if match_name:
            return match_name.group(1)
        return val.split("/")[0].strip()
    return "Unknown"

df["category_clean"] = df["category"].apply(parse_category)

# Group rare categories
top_cats = df["category_clean"].value_counts().nlargest(15).index
df["category_clean"] = df["category_clean"].apply(lambda x: x if x in top_cats else "Other")

# Country grouping
top_countries = df["country"].value_counts().nlargest(10).index
df["country_clean"] = df["country"].apply(lambda x: x if x in top_countries else "Other")

# ==========================================
# 4. FEATURE SELECTION & SPLIT
# ==========================================
numeric_features = [
    "log_goal_usd",
    "campaign_duration_days",
    "prep_duration_days",
    "launch_hour",
    "name_len",
    "name_word_count",
    "blurb_len",
    "blurb_word_count",
    "has_video",
    "staff_pick",
    "prelaunch_activated",
]

categorical_features = [
    "category_clean",
    "country_clean",
    "launch_month",
    "launch_day_of_week",
]

all_features = numeric_features + categorical_features
X = df[all_features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train Set: {X_train.shape[0]:,} records | Test Set: {X_test.shape[0]:,} records")

# ==========================================
# 5. PREPROCESSING PIPELINE
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# ==========================================
# 6. MODEL TRAINING & COMPARISON
# ==========================================
models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost (Champion)": XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1
    ),
}

results = []
trained_pipelines = {}

print("\n" + "=" * 60)
print("Training models...")

for name, clf in models.items():
    print(f"--> Training {name}...")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Predictions
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)

    results.append({
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Brier Score": round(brier, 4),
    })

# ==========================================
# 7. RESULTS SUMMARY
# ==========================================
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL BENCHMARK RESULTS (Academic & Demo Verification):")
print(results_df.to_string(index=False))

# Save the champion model (XGBoost) for Streamlit demo
champion_pipeline = trained_pipelines["XGBoost (Champion)"]
joblib.dump(champion_pipeline, "projectiq_model.joblib")
joblib.dump(all_features, "model_features.joblib")
print("\nChampion model saved successfully as 'projectiq_model.joblib'!")
print("=" * 60)
