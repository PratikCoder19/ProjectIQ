"""
ProjectIQ V2.0 — Core Prediction & Risk Calibration Module
Encapsulates T_0 pre-launch inference and risk tier mapping.
"""

import joblib
import numpy as np
import pandas as pd

RISK_THRESHOLDS = {
    "LOW": 35.0,     # Risk score < 35% -> Low Risk
    "MEDIUM": 65.0   # 35% <= Risk score <= 65% -> Medium Risk
                     # Risk score > 65% -> High Risk
}

class ProjectIQPredictor:
    def __init__(self, model_path: str = "projectiq_model.joblib", features_path: str = "model_features.joblib"):
        self.pipeline = joblib.load(model_path)
        self.expected_features = joblib.load(features_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

    def prepare_input(self, raw_input: dict) -> pd.DataFrame:
        """
        Transforms raw user input into the strict pre-launch feature matrix.
        Enforces zero target leakage.
        """
        goal = float(raw_input.get("goal_usd", 10000))
        log_goal = np.log1p(goal)
        
        name = str(raw_input.get("name", ""))
        blurb = str(raw_input.get("blurb", ""))
        
        df = pd.DataFrame([{
            "log_goal_usd": log_goal,
            "campaign_duration_days": float(raw_input.get("campaign_duration_days", 30)),
            "prep_duration_days": float(raw_input.get("prep_duration_days", 30)),
            "launch_hour": int(raw_input.get("launch_hour", 14)),
            "name_len": len(name),
            "name_word_count": len(name.split()),
            "blurb_len": len(blurb),
            "blurb_word_count": len(blurb.split()),
            "has_video": int(raw_input.get("has_video", 1)),
            "staff_pick": int(raw_input.get("staff_pick", 0)),
            "prelaunch_activated": int(raw_input.get("prelaunch_activated", 1)),
            "category_clean": str(raw_input.get("category_clean", "Other")),
            "country_clean": str(raw_input.get("country_clean", "US")),
            "launch_month": str(raw_input.get("launch_month", "8")),
            "launch_day_of_week": str(raw_input.get("launch_day_of_week", "Tuesday"))
        }])
        return df[self.expected_features]

    def predict(self, raw_input: dict) -> dict:
        """
        Performs inference and returns probability, calibrated risk score, and risk tier.
        """
        input_df = self.prepare_input(raw_input)
        prob_success = float(self.pipeline.predict_proba(input_df)[0, 1])
        risk_score = (1.0 - prob_success) * 100.0
        
        if risk_score < RISK_THRESHOLDS["LOW"]:
            risk_tier = "LOW"
        elif risk_score <= RISK_THRESHOLDS["MEDIUM"]:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"
            
        return {
            "success_probability": prob_success,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "input_df": input_df
        }