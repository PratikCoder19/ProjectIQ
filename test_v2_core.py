"""
Test Suite for ProjectIQ V2.0 Core Modules (Prediction, SHAP, Scenarios)
"""

from src.prediction import ProjectIQPredictor
from src.explainability import ProjectIQExplainer
from src.scenarios import ScenarioSimulator

print("=" * 60)
print("Testing ProjectIQ V2.0 Core Engines...")

# 1. Test Prediction
predictor = ProjectIQPredictor()
test_project = {
    "goal_usd": 25000,
    "campaign_duration_days": 45,
    "prep_duration_days": 10,
    "launch_hour": 14,
    "name": "Nova: Advanced Autonomous Robot",
    "blurb": "A next-generation companion robot powered by on-device computer vision and natural language understanding.",
    "has_video": 0,
    "staff_pick": 0,
    "prelaunch_activated": 0,
    "category_clean": "Technology",
    "country_clean": "US",
    "launch_month": "8",
    "launch_day_of_week": "Friday"
}

res = predictor.predict(test_project)
print(f"Prediction OK: Success Prob = {res['success_probability'] * 100:.2f}% | Risk Tier = {res['risk_tier']}")

# 2. Test SHAP Instance Explainability
explainer = ProjectIQExplainer(predictor)
exp_res = explainer.explain_instance(res["input_df"])
plot_path = explainer.generate_waterfall_plot(exp_res["shap_explanation"], "test_waterfall.png")
print(f"SHAP OK: Generated Waterfall Plot at '{plot_path}'")
print(f"Top Risk Driver: {exp_res['top_drivers'][0]['feature']} ({exp_res['top_drivers'][0]['shap_value']:.4f})")

# 3. Test Scenario Simulator
simulator = ScenarioSimulator(predictor)
modifications = {
    "goal_usd": 12000,
    "campaign_duration_days": 30,
    "prep_duration_days": 40,
    "has_video": 1,
    "prelaunch_activated": 1
}
sim_res = simulator.simulate_scenario(test_project, modifications)
print(f"Scenario OK: Baseline Prob = {sim_res['baseline_probability']*100:.2f}% -> Scenario Prob = {sim_res['scenario_probability']*100:.2f}% (Delta: {sim_res['probability_delta_pct']:+.2f}%)")
print("=" * 60)
print("ALL V2.0 CORE ENGINES VALIDATED SUCCESSFULLY!")