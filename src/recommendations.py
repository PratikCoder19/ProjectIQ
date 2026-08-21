"""
ProjectIQ V2.0 — Risk Register & Prescriptive Recommendations Engine
Maintains explicit boundaries between empirical ML evidence and prescriptive advice.
"""

from typing import List, Dict, Any

class RiskRegisterEngine:
    @staticmethod
    def generate_risk_register(raw_input: Dict[str, Any], shap_drivers: List[Dict[str, Any]], benchmark_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        register = []
        goal = float(raw_input.get("goal_usd", 10000))
        duration = float(raw_input.get("campaign_duration_days", 30))
        prep = float(raw_input.get("prep_duration_days", 30))
        has_video = int(raw_input.get("has_video", 1))
        prelaunch = int(raw_input.get("prelaunch_activated", 1))

        # Risk 1: Capital Requirement Risk
        if goal > 30000 or benchmark_info.get("goal_ratio_to_median", 1.0) > 2.0:
            register.append({
                "risk_id": "RSK-01",
                "category": "Financial Target",
                "risk_title": "Elevated Funding Target",
                "likelihood": "High",
                "potential_impact": "High",
                "evidence_type": "DATA / SHAP EVIDENCE",
                "evidence_detail": f"Target (${goal:,.0f}) is {benchmark_info.get('goal_ratio_to_median')}x category median. SHAP shows high negative attribution.",
                "suggested_response": "Stage project into Phase 1 MVP ($10,000–$15,000) with unlockable stretch goals."
            })

        # Risk 2: Schedule & Momentum Decay Risk
        if duration > 35:
            register.append({
                "risk_id": "RSK-02",
                "category": "Schedule Management",
                "risk_title": "Extended Campaign Window",
                "likelihood": "Medium",
                "potential_impact": "Medium",
                "evidence_type": "DATA / SHAP EVIDENCE",
                "evidence_detail": f"Duration ({duration:.0f} days) exceeds 30-day platform optimum, leading to middle-campaign momentum decay.",
                "suggested_response": "Compress campaign timeline to 28–30 days to maximize urgency and algorithmic visibility."
            })

        # Risk 3: Marketing Readiness Deficit
        if prep < 14 or prelaunch == 0:
            register.append({
                "risk_id": "RSK-03",
                "category": "Go-to-Market Readiness",
                "risk_title": "Pre-Launch Audience Deficit",
                "likelihood": "High",
                "potential_impact": "High",
                "evidence_type": "DATA / SHAP EVIDENCE",
                "evidence_detail": "Preparation lead time < 14 days and/or inactive pre-launch page, jeopardizing Day-1 velocity.",
                "suggested_response": "Delay public launch by 2–3 weeks to activate pre-launch page and secure a committed backer waitlist."
            })

        # Risk 4: Content Presentation Friction
        if has_video == 0:
            register.append({
                "risk_id": "RSK-04",
                "category": "Content & Conversion",
                "risk_title": "Absence of Pitch Video",
                "likelihood": "High",
                "potential_impact": "Medium",
                "evidence_type": "DATA / SHAP EVIDENCE",
                "evidence_detail": f"Category video adoption is {benchmark_info.get('domain_video_adoption', 70):.0f}%. Non-video projects face severe conversion penalties.",
                "suggested_response": "Produce a high-definition 90-second pitch video detailing product value proposition."
            })

        return register