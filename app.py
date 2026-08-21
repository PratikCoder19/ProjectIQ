"""
ProjectIQ V2.0 — AI-Driven Project Success & Risk Decision Support System
Integrated Decision Intelligence Dashboard
Prediction Point: T₀ (Pre-Launch Lock)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Import modular V2.0 decision intelligence engines
from src.prediction import ProjectIQPredictor
from src.explainability import ProjectIQExplainer
from src.scenarios import ScenarioSimulator
from src.benchmarking import ProjectIQBenchmarker
from src.recommendations import RiskRegisterEngine
from src.llm import AICopilot
from src.reporting import DecisionReportGenerator

st.set_page_config(
    page_title="ProjectIQ V2.0 | Decision Support System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_engines():
    pred = ProjectIQPredictor()
    exp = ProjectIQExplainer(pred)
    sim = ScenarioSimulator(pred)
    bm = ProjectIQBenchmarker()
    copilot = AICopilot()
    return pred, exp, sim, bm, copilot

try:
    predictor, explainer, simulator, benchmarker, ai_copilot = initialize_engines()
except Exception as e:
    st.error(f"Error initializing ProjectIQ V2.0 engines: {e}")
    st.stop()

# ==========================================
# SIDEBAR: AI ENGINE & PROJECT INPUTS
# ==========================================
with st.sidebar:
    st.header("🤖 AI Copilot Model Status")
    
    model_provider_choice = st.selectbox(
        "Select Active AI Engine",
        ["Auto-Detect", "Google Gemini", "OpenAI", "Groq Cloud", "Local Ollama", "Deterministic (No LLM)"]
    )
    
    provider_map = {
        "Auto-Detect": "auto",
        "Google Gemini": "gemini",
        "OpenAI": "openai",
        "Groq Cloud": "groq",
        "Local Ollama": "ollama",
        "Deterministic (No LLM)": "deterministic"
    }
    ai_copilot.set_provider(provider_map[model_provider_choice])
    
    status_info = ai_copilot.get_active_provider_status()
    st.success(f"**Connected Model:**\n\n`{status_info['active_model']}`")
    st.caption(f"Engine Type: **{status_info['engine_type']}**")
    
    st.markdown("---")
    st.header("📋 Project Configuration (T₀)")
    
    st.subheader("1. General & Financial")
    proj_name = st.text_input("Project Name", value="Nova: Autonomous AI Companion")
    proj_blurb = st.text_area(
        "Pitch Blurb",
        value="A next-generation companion robot powered by on-device computer vision and natural language understanding."
    )
    goal_usd = st.number_input("Target Goal (USD $)", min_value=100, max_value=10000000, value=25000, step=500)
    
    st.subheader("2. Domain & Market")
    category_options = [
        "Technology", "Product Design", "Tabletop Games", "Video Games", "Film & Video",
        "Music", "Publishing", "Art", "Fashion", "Food", "Crafts", "Other"
    ]
    category = st.selectbox("Category Domain", category_options, index=0)
    
    # Full country names mapping to model code
    country_display_map = {
        "United States": "US",
        "United Kingdom": "GB",
        "Canada": "CA",
        "Australia": "AU",
        "Germany": "DE",
        "France": "FR",
        "Italy": "IT",
        "Netherlands": "NL",
        "Spain": "ES",
        "Sweden": "SE",
        "Other / International": "Other"
    }
    selected_country_name = st.selectbox("Target Country", list(country_display_map.keys()), index=0)
    country_code = country_display_map[selected_country_name]
    
    st.subheader("3. Schedule & Timing")
    campaign_duration = st.slider("Campaign Duration (Days)", min_value=7, max_value=90, value=45)
    prep_duration = st.slider("Preparation Window (Days)", min_value=1, max_value=365, value=10)
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        launch_month = st.selectbox("Launch Month", [str(i) for i in range(1, 13)], index=7)
    with c_m2:
        launch_day = st.selectbox("Launch Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    launch_hour = st.slider("Launch Hour (UTC)", 0, 23, 14)
    
    st.subheader("4. Assets & Readiness Signals")
    # Tooltips provided on hover
    has_video = st.checkbox(
        "Pitch Video Included",
        value=False,
        help="Projects featuring a video pitch provide higher social proof and historically experience significantly stronger conversion and trust."
    )
    prelaunch_activated = st.checkbox(
        "Pre-Launch Page Active",
        value=False,
        help="An active pre-launch landing page captures an early waitlist, driving essential Day-1 funding momentum and algorithmic visibility."
    )
    staff_pick = st.checkbox(
        "Platform Staff Pick",
        value=False,
        help="Indicates editorial platform curation or featured placement, substantially increasing organic discovery across backer communities."
    )

st.title("🚀 ProjectIQ V2.0: AI Decision Intelligence System")
st.markdown(
    f"**Active Prediction Engine:** `XGBoost ML Pipeline (AUC: 0.8646)` | **Active AI Advisory:** `{status_info['active_model']}`"
)
st.markdown("---")

baseline_input = {
    "name": proj_name,
    "blurb": proj_blurb,
    "goal_usd": goal_usd,
    "campaign_duration_days": campaign_duration,
    "prep_duration_days": prep_duration,
    "launch_hour": launch_hour,
    "has_video": int(has_video),
    "staff_pick": int(staff_pick),
    "prelaunch_activated": int(prelaunch_activated),
    "category_clean": category,
    "country_clean": country_code,
    "launch_month": launch_month,
    "launch_day_of_week": launch_day
}

pred_result = predictor.predict(baseline_input)
shap_result = explainer.explain_instance(pred_result["input_df"])
benchmark_info = benchmarker.get_benchmark(category, goal_usd, campaign_duration)
risk_register = RiskRegisterEngine.generate_risk_register(baseline_input, shap_result["top_drivers"], benchmark_info)

tab_assess, tab_shap, tab_sim, tab_bench, tab_reg, tab_copilot, tab_report = st.tabs([
    "🎯 1. Project Assessment",
    "🔍 2. SHAP Explainability",
    "⚡ 3. What-If Simulator",
    "📊 4. Benchmarking",
    "🛡️ 5. Risk Register",
    "🤖 6. AI Copilot Brief",
    "📄 7. Decision Report"
])

# ------------------------------------------
# TAB 1: PROJECT ASSESSMENT
# ------------------------------------------
with tab_assess:
    st.subheader("Project Outcome Probability & Calibrated Risk Tier")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(
            label="Estimated Success Probability",
            value=f"{pred_result['success_probability'] * 100:.1f}%",
            delta=f"{(pred_result['success_probability'] - 0.6322) * 100:+.1f}% vs Platform Avg"
        )
    with col_kpi2:
        tier = pred_result["risk_tier"]
        color = "🟢" if tier == "LOW" else ("🟡" if tier == "MEDIUM" else "🔴")
        st.metric(label="Overall Risk Tier", value=f"{color} {tier} RISK")
    with col_kpi3:
        st.metric(label="Calibrated Risk Score", value=f"{pred_result['risk_score']:.1f} / 100")
        
    st.progress(float(pred_result["success_probability"]))
    st.info("💡 **Academic Note:** Probabilities are calibrated on unseen holdout data ($N=34,049$, Brier Score: $0.1411$). Predictions represent statistical associations at $T_0$ and do not imply causal certainty.")

# ------------------------------------------
# TAB 2: SHAP EXPLAINABILITY
# ------------------------------------------
with tab_shap:
    st.subheader("Instance-Level Factor Attribution (Local TreeSHAP)")
    
    waterfall_path = "instance_shap_waterfall.png"
    pie_path = "instance_shap_pie.png"
    
    explainer.generate_waterfall_plot(shap_result["shap_explanation"], waterfall_path)
    explainer.generate_pie_chart(shap_result["pos_impact_sum"], shap_result["neg_impact_sum"], pie_path)
    
    col_waterfall, col_pie = st.columns([1.3, 1])
    with col_waterfall:
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption="SHAP Waterfall: How specific project features shift base log-odds to the prediction.")
    with col_pie:
        if os.path.exists(pie_path):
            st.image(pie_path, caption="Factor Balance: Total volume of Risk-Increasing vs Success-Supporting factors.")
            
    st.markdown("##### Detailed Attribution Breakdown")
    col_drivers, col_expl = st.columns([1.2, 1])
    with col_drivers:
        for d in shap_result["top_drivers"][:6]:
            icon = "✅" if d["shap_value"] > 0 else "⚠️"
            st.write(f"{icon} **{d['feature']}**: `{d['shap_value']:+.3f}` ({d['direction']})")
    with col_expl:
        st.caption(
            "• **Red/Negative Factors:** Increase the likelihood of funding failure relative to platform average.\n"
            "• **Green/Positive Factors:** Exert upward force driving project toward successful funding."
        )

# ------------------------------------------
# TAB 3: WHAT-IF SIMULATOR
# ------------------------------------------
with tab_sim:
    st.subheader("⚡ Model-Based What-If Scenario Simulator")
    st.caption("Simulate how altering controllable decisions shifts model probability. (Non-causal sensitivity analysis)")
    
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        sim_goal = st.number_input("Simulated Goal ($)", min_value=100, max_value=1000000, value=int(min(goal_usd, 12000)), step=500)
    with c_s2:
        sim_dur = st.slider("Simulated Duration (Days)", 7, 90, value=int(min(campaign_duration, 30)))
    with c_s3:
        sim_prep = st.slider("Simulated Prep (Days)", 1, 365, value=int(max(prep_duration, 35)))
        
    c_s4, c_s5 = st.columns(2)
    with c_s4:
        sim_video = st.checkbox("Simulate With Pitch Video", value=True)
    with c_s5:
        sim_prelaunch = st.checkbox("Simulate With Pre-Launch Page", value=True)
        
    mod_params = {
        "goal_usd": sim_goal,
        "campaign_duration_days": sim_dur,
        "prep_duration_days": sim_prep,
        "has_video": int(sim_video),
        "prelaunch_activated": int(sim_prelaunch)
    }
    
    sim_res = simulator.simulate_scenario(baseline_input, mod_params)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Baseline Success Prob", f"{sim_res['baseline_probability'] * 100:.1f}%")
    with col_r2:
        st.metric("Scenario Success Prob", f"{sim_res['scenario_probability'] * 100:.1f}%")
    with col_r3:
        st.metric("Projected Probability Shift", f"{sim_res['probability_delta_pct']:+.1f}%", delta=f"{sim_res['probability_delta_pct']:+.1f}%")

# ------------------------------------------
# TAB 4: BENCHMARKING
# ------------------------------------------
with tab_bench:
    st.subheader(f"📊 Domain Benchmarks: {category}")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Category Success Rate", f"{benchmark_info['domain_success_rate']:.1f}%")
    with b2:
        st.metric("Domain Median Goal", f"${benchmark_info['domain_median_goal']:,}")
    with b3:
        st.metric("Domain Median Duration", f"{benchmark_info['domain_median_duration']} Days")
    with b4:
        st.metric("Video Adoption Rate", f"{benchmark_info['domain_video_adoption']:.0f}%")
    st.write(f"**Goal Calibration Assessment:** `{benchmark_info['goal_assessment']}` ({benchmark_info['goal_ratio_to_median']}x Category Median)")

# ------------------------------------------
# TAB 5: RISK REGISTER
# ------------------------------------------
with tab_reg:
    st.subheader("🛡️ Project Risk Register & Action Plan")
    st.caption("Distinguishing empirical ML/SHAP data evidence from prescriptive management responses.")
    reg_df = pd.DataFrame(risk_register)
    if not reg_df.empty:
        st.dataframe(
            reg_df[["risk_id", "category", "risk_title", "likelihood", "potential_impact", "evidence_detail", "suggested_response"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No critical risk factors identified under current project parameters.")

# ------------------------------------------
# TAB 6: AI COPILOT BRIEF
# ------------------------------------------
with tab_copilot:
    st.subheader("🤖 ProjectIQ AI Copilot: Executive Risk Advisory")
    context_payload = {
        "name": proj_name,
        "success_probability": pred_result["success_probability"] * 100.0,
        "risk_tier": pred_result["risk_tier"],
        "goal_usd": goal_usd,
        "benchmark_median_goal": benchmark_info["domain_median_goal"],
        "duration_days": campaign_duration,
        "prep_days": prep_duration,
        "top_risk_drivers": shap_result["top_drivers"],
        "risk_register": risk_register,
        "scenario_summary": f"Simulated optimization suggests a potential shift to {sim_res['scenario_probability'] * 100:.1f}%."
    }
    copilot_output = ai_copilot.generate_narrative_brief(context_payload)
    st.info(f"Generated by: **{copilot_output['source']}**")
    st.markdown(copilot_output["content"])

# ------------------------------------------
# TAB 7: DECISION REPORT
# ------------------------------------------
with tab_report:
    st.subheader("📄 Export Executive Decision Brief")
    if st.button("Generate Official Decision Brief (PDF)", type="primary"):
        pdf_payload = {
            "name": proj_name,
            "category": category,
            "goal_usd": goal_usd,
            "duration_days": campaign_duration,
            "success_probability": pred_result["success_probability"] * 100.0,
            "risk_tier": pred_result["risk_tier"],
            "risk_register": risk_register,
            "narrative": copilot_output["content"]
        }
        pdf_file = DecisionReportGenerator.generate_pdf(pdf_payload, "ProjectIQ_Decision_Brief.pdf")
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download PDF Decision Brief",
                data=f,
                file_name="ProjectIQ_Decision_Brief.pdf",
                mime="application/pdf"
            )
        st.success("Decision brief compiled.")

st.markdown("---")
st.caption("ProjectIQ V2.0 Architecture: Pre-Launch Feature Pipeline → Calibrated XGBoost ML Engine → SHAP Explainability → Decision Intelligence → PDF Generator")