"""
ProjectIQ V2.0 — Resilient AI Copilot Provider Interface
Supports OpenAI API, Local Ollama, or deterministic rule-based fallback.
"""

import json
import os
from typing import Dict, Any

class AICopilot:
    def __init__(self, provider: str = "auto", api_key: str = None, model_name: str = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    def generate_narrative_brief(self, context_payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates structured executive interpretation and actionable recommendations.
        Falls back seamlessly to deterministic generation if no LLM service is available.
        """
        # Try OpenAI if configured
        if self.provider in ["openai", "auto"] and self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = self._build_prompt(context_payload)
                
                resp = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are ProjectIQ AI, a senior project risk consultant. Synthesize the provided quantitative data into a concise executive brief. Do not invent statistics or assume causality."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=450
                )
                return {
                    "source": "OpenAI LLM Engine",
                    "content": resp.choices[0].message.content.strip()
                }
            except Exception as e:
                pass  # Gracefully proceed to fallback

        # Try Local Ollama if configured
        if self.provider in ["ollama", "auto"]:
            try:
                import urllib.request
                prompt = self._build_prompt(context_payload)
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps({"model": self.model_name or "tinyllama", "prompt": prompt, "stream": False}).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=3) as response:
                    res_data = json.loads(response.read().decode())
                    return {
                        "source": "Local Ollama LLM",
                        "content": res_data.get("response", "").strip()
                    }
            except Exception:
                pass  # Gracefully proceed to fallback

        # Deterministic Grounded Engine (Always Available)
        return {
            "source": "ProjectIQ Deterministic Intelligence Engine",
            "content": self._build_deterministic_narrative(context_payload)
        }

    def _build_prompt(self, p: Dict[str, Any]) -> str:
        return f"""
Analyze the following project risk data:
- Project Name: {p.get('name')}
- Success Probability: {p.get('success_probability'):.1f}% (Risk Tier: {p.get('risk_tier')})
- Target Goal: ${p.get('goal_usd'):,.0f} (Category Median: ${p.get('benchmark_median_goal'):,.0f})
- Duration: {p.get('duration_days')} days | Prep Window: {p.get('prep_days')} days
- Top Risk Factors (SHAP): {', '.join([d['feature'] for d in p.get('top_risk_drivers', [])])}
- Scenario Impact: {p.get('scenario_summary', 'N/A')}

Produce 3 concise sections:
1. Executive Risk Summary
2. Top Vulnerabilities
3. Recommended Managerial Interventions
"""

    def _build_deterministic_narrative(self, p: Dict[str, Any]) -> str:
        prob = p.get('success_probability', 50.0)
        tier = p.get('risk_tier', 'MEDIUM')
        goal = p.get('goal_usd', 10000)
        
        narrative = f"### Executive Summary\n"
        narrative += f"The project demonstrates a **{prob:.1f}% estimated success probability**, placing it in the **{tier} RISK** category. "
        
        if tier == "HIGH":
            narrative += f"The primary exposure stems from an aggressive funding goal (${goal:,.0f}) and insufficient pre-launch lead time. "
        elif tier == "LOW":
            narrative += f"The project's parameters are well-calibrated against historical top-quartile performers. "
        else:
            narrative += f"The project exhibits moderate viability with specific optimization opportunities in timeline and marketing preparation. "

        narrative += f"\n\n### Strategic Interventions\n"
        for idx, rsk in enumerate(p.get("risk_register", [])[:3], 1):
            narrative += f"{idx}. **{rsk['risk_title']}**: {rsk['suggested_response']}\n"

        narrative += "\n*Note: Recommendations are model-based decision-support guidelines derived from empirical platform patterns.*"
        return narrative