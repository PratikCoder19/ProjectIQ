"""
ProjectIQ v0.1 — AI-Driven Project Success & Risk Decision Support System
Demo Target: 17 August Faculty Review
Stack: Streamlit + Scikit-Learn / XGBoost + Prescriptive Logic
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="ProjectIQ | Decision Support System",
    page_icon="📊",
    layout="wide"
)

# Load trained pipeline and features
@st.cache_resource
def load_assets():
    model = joblib.load("projectiq_model.joblib")
    features = joblib.load("model_features.joblib")
    return model, features

try:
    model_pipeline, feature_cols = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure 'pipeline.py' was executed successfully.")
    st.stop()

# Header & Academic Context
st.title("🚀 ProjectIQ: AI Project Risk & Decision Support System")
st.caption("PGDM Data Science Dissertation Prototype | Prediction Point: T₀ (Pre-Launch Lock)")
st.markdown("---")

# Layout: 2 Columns (Input Panel vs. Analytics & Decision Output)
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Project Parameters (Pre-Launch)")
    
    # Financial Inputs
    st.markdown("##### 1. Financial & Goal Parameters")
    goal_usd = st.number_input("Target Funding Goal (USD $)", min_value=100, max_value=10000000, value=15000, step=500)
    
    # Category & Geography
    st.markdown("##### 2. Domain & Market")
    category_options = [
        "Product Design", "Tabletop Games", "Video Games", "Shorts", "Documentary",
        "Fiction", "Fashion", "Art", "Technology", "Theater", "Music", "Publishing",
        "Film & Video", "Food", "Comics", "Other"
    ]
    category = st.selectbox("Project Category", category_options)
    
    country_options = ["US", "GB", "CA", "AU", "DE", "FR", "IT", "NL", "ES", "SE", "Other"]
    country = st.selectbox("Target Country", country_options)
    
    # Schedule & Timing
    st.markdown("##### 3. Timeline & Schedule")
    campaign_duration = st.slider("Campaign Duration (Days)", min_value=7, max_value=90, value=30)
    prep_duration = st.slider("Preparation Window (Days from Draft to Launch)", min_value=1, max_value=365, value=45)
    
    c_time1, c_time2, c_time3 = st.columns(3)
    with c_time1:
        launch_month = st.selectbox("Launch Month", [str(i) for i in range(1, 13)], index=7)
    with c_time2:
        launch_day = st.selectbox("Launch Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=1)
    with c_time3:
        launch_hour = st.slider("Launch Hour (UTC)", 0, 23, 14)
        
    # Content & Readiness Signals
    st.markdown("##### 4. Content Quality & Readiness Signals")
    name_text = st.text_input("Project Title / Headline", value="SmartPulse: Next-Gen Autonomous Fitness Tracker")
    blurb_text = st.text_area("Campaign Blurb / Summary", value="An AI-powered wearable that tracks fatigue, posture, and recovery in real-time with 7-day battery life.")
    
    c_sig1, c_sig2, c_sig3 = st.columns(3)
    with c_sig1:
        has_video = st.checkbox("Video Pitch Included", value=True)
    with c_sig2:
        prelaunch_active = st.checkbox("Pre-Launch Page Active", value=True)
    with c_sig3:
        staff_pick = st.checkbox("Staff Pick / Featured", value=False)

# Compute derived inputs
log_goal = np.log1p(goal_usd)
name_len = len(name_text)
name_words = len(name_text.split())
blurb_len = len(blurb_text)
blurb_words = len(blurb_text.split())

input_data = pd.DataFrame([{
    "log_goal_usd": log_goal,
    "campaign_duration_days": campaign_duration,
    "prep_duration_days": prep_duration,
    "launch_hour": launch_hour,
    "name_len": name_len,
    "name_word_count": name_words,
    "blurb_len": blurb_len,
    "blurb_word_count": blurb_words,
    "has_video": int(has_video),
    "staff_pick": int(staff_pick),
    "prelaunch_activated": int(prelaunch_active),
    "category_clean": category,
    "country_clean": country,
    "launch_month": launch_month,
    "launch_day_of_week": launch_day
}])

# Inference
with col_output:
    st.subheader("🎯 Decision Support & Risk Analysis")
    
    # Model probabilities
    prob_success = model_pipeline.predict_proba(input_data)[0, 1]
    risk_score = (1.0 - prob_success) * 100
    
    # KPI Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Predicted Success Probability",
            value=f"{prob_success * 100:.1f}%",
            delta=f"{(prob_success - 0.6322) * 100:+.1f}% vs Baseline"
        )
    with m2:
        risk_label = "LOW" if risk_score < 30 else ("MEDIUM" if risk_score < 60 else "HIGH")
        st.metric(label="Overall Project Risk Tier", value=risk_label)
    with m3:
        st.metric(label="Calibrated Risk Score", value=f"{risk_score:.1f} / 100")
        
    st.progress(float(prob_success))
    st.markdown("---")
    
    # Explainable Risk Factor Assessment
    st.subheader("🔍 Major Risk & Success Drivers")
    
    driver_factors = []
    
    # Goal-based rules
    if goal_usd > 50000:
        driver_factors.append(("⚠️ High Funding Goal ($>50k)", "Capital requirement is significantly above platform median, increasing failure risk.", "negative"))
    elif goal_usd <= 10000:
        driver_factors.append(("✅ Optimal Funding Goal", "Goal size aligns with historical high-conversion campaigns.", "positive"))
        
    # Duration rules
    if campaign_duration > 35:
        driver_factors.append(("⚠️ Extended Campaign Duration (>35 Days)", "Longer campaigns experience momentum decay and donor fatigue.", "negative"))
    elif 25 <= campaign_duration <= 35:
        driver_factors.append(("✅ Optimal Duration Window", "30-day target maximizes urgency while allowing organic reach.", "positive"))
        
    # Video & Content rules
    if not has_video:
        driver_factors.append(("❌ Missing Video Asset", "Projects without pitch videos historically suffer severe conversion drops.", "negative"))
    else:
        driver_factors.append(("✅ Video Pitch Present", "Strong visual asset increases creator credibility.", "positive"))
        
    if not prelaunch_active:
        driver_factors.append(("⚠️ No Pre-Launch Page", "Lack of pre-launch lead generation reduces day-1 funding velocity.", "negative"))
    else:
        driver_factors.append(("✅ Pre-Launch Validation", "Active pre-launch page builds Day-1 donor momentum.", "positive"))
        
    if prep_duration < 14:
        driver_factors.append(("⚠️ Short Preparation Window (<14 Days)", "Fast turnarounds correlate with incomplete marketing assets.", "negative"))

    for title, desc, sentiment in driver_factors:
        if sentiment == "positive":
            st.success(f"**{title}**: {desc}")
        else:
            st.warning(f"**{title}**: {desc}")
            
    st.markdown("---")
    
    # Actionable Managerial Recommendations
    st.subheader("💡 Prescriptive Management Recommendations")
    
    recs = []
    if goal_usd > 30000:
        recs.append("• **De-risk Funding Target:** Consider staging the release into Phase 1 (MVP) with a lower initial target ($15,000–$20,000) and stretch goals.")
    if campaign_duration > 35:
        recs.append(f"• **Shorten Schedule:** Reduce duration from {campaign_duration} days to 30 days to sustain backer urgency and algorithm visibility.")
    if not has_video:
        recs.append("• **Produce High-Definition Pitch Video:** Essential to improve conversion rates and qualify for staff curation.")
    if not prelaunch_active:
        recs.append("• **Activate Pre-Launch Waitlist:** Run a 30-day pre-launch landing page to guarantee at least 20% of goal funded within the first 48 hours.")
    if blurb_words < 12 or blurb_words > 25:
        recs.append(f"• **Optimize Copy Length:** Current blurb is {blurb_words} words. Target the platform sweet spot of 15–22 descriptive words.")
        
    if not recs:
        recs.append("• **All parameters optimized:** Project configuration aligns with top-quartile historical performers. Proceed with promotional execution.")
        
    for r in recs:
        st.write(r)

st.markdown("---")
st.caption("ProjectIQ Architecture: Data Pipeline → Pre-Launch Feature Store → XGBoost ML Engine (AUC: 0.8646) → Prescriptive Decision Layer")
