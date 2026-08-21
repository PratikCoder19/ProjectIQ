"""
ProjectIQ V2.0 — What-If Scenario Simulation Engine
Evaluates sensitivity under modified controllable parameters.
"""

from typing import Dict, Any

CONTROLLABLE_ATTRIBUTES = [
    "goal_usd",
    "campaign_duration_days",
    "prep_duration_days",
    "has_video",
    "prelaunch_activated"
]

class ScenarioSimulator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_scenario(self, baseline_input: Dict[str, Any], modified_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs sensitivity analysis by modifying only controllable parameters.
        """
        # Create modified scenario payload
        scenario_input = baseline_input.copy()
        for key, value in modified_params.items():
            if key in CONTROLLABLE_ATTRIBUTES:
                scenario_input[key] = value

        # Run inference on baseline and scenario
        base_res = self.predictor.predict(baseline_input)
        scen_res = self.predictor.predict(scenario_input)

        prob_delta = (scen_res["success_probability"] - base_res["success_probability"]) * 100.0
        risk_delta = (scen_res["risk_score"] - base_res["risk_score"])

        # Document specific changes
        changes = {}
        for key in CONTROLLABLE_ATTRIBUTES:
            if key in modified_params and modified_params[key] != baseline_input.get(key):
                changes[key] = {
                    "original": baseline_input.get(key),
                    "scenario": modified_params[key]
                }

        return {
            "baseline_probability": base_res["success_probability"],
            "scenario_probability": scen_res["success_probability"],
            "probability_delta_pct": prob_delta,
            "baseline_risk_score": base_res["risk_score"],
            "scenario_risk_score": scen_res["risk_score"],
            "risk_delta": risk_delta,
            "scenario_risk_tier": scen_res["risk_tier"],
            "modified_parameters": changes,
            "disclaimer": "Model-based sensitivity simulation under modified inputs; does not establish causal certainty."
        }