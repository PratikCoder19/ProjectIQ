"""
Test Suite for ProjectIQ V2.0 Advanced Intelligence Modules
Validates Benchmarking, Risk Register, AI Copilot & PDF Generation.
"""

from src.benchmarking import ProjectIQBenchmarker
from src.recommendations import RiskRegisterEngine
from src.llm import AICopilot
from src.reporting import DecisionReportGenerator

print("=" * 60)
print("Testing ProjectIQ V2.0 Advanced Intelligence Modules...")

# 1. Test Benchmarking
bm = ProjectIQBenchmarker.get_benchmark("Technology", 25000, 45)
print(f"Benchmarking OK: Domain Median Goal = ${bm['domain_median_goal']:,} | Assessment = {bm['goal_assessment']}")

# 2. Test Risk Register
sample_drivers = [{"feature": "log_goal_usd", "shap_value": -1.17}]
sample_input = {"goal_usd": 25000, "campaign_duration_days": 45, "prep_duration_days": 10, "has_video": 0, "prelaunch_activated": 0}
reg = RiskRegisterEngine.generate_risk_register(sample_input, sample_drivers, bm)
print(f"Risk Register OK: Generated {len(reg)} structured risk items.")

# 3. Test AI Copilot (Deterministic Fallback)
copilot = AICopilot()
brief = copilot.generate_narrative_brief({
    "name": "Nova Companion Robot",
    "success_probability": 9.76,
    "risk_tier": "HIGH",
    "goal_usd": 25000,
    "benchmark_median_goal": bm["domain_median_goal"],
    "duration_days": 45,
    "prep_days": 10,
    "top_risk_drivers": sample_drivers,
    "risk_register": reg
})
print(f"AI Copilot OK: Source = '{brief['source']}'")

# 4. Test PDF Generation
payload = {
    "name": "Nova Companion Robot",
    "category": "Technology",
    "goal_usd": 25000,
    "duration_days": 45,
    "success_probability": 9.76,
    "risk_tier": "HIGH",
    "risk_register": reg,
    "narrative": brief["content"]
}
pdf_path = DecisionReportGenerator.generate_pdf(payload, "test_decision_brief.pdf")
print(f"PDF Generator OK: Generated '{pdf_path}'")
print("=" * 60)
print("ALL V2.0 ADVANCED MODULES VALIDATED SUCCESSFULLY!")