"""
ProjectIQ V2.0 — Multi-Provider AI Copilot Interface
Updated for robust local Ollama inference with proper timeout & model resolution.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Dict, Any

class AICopilot:
    def __init__(self, provider: str = "auto", model_name: str = None):
        self.provider = provider
        self.model_name = model_name

    def set_provider(self, provider: str, model_name: str = None):
        self.provider = provider
        self.model_name = model_name

    def get_active_provider_status(self) -> Dict[str, Any]:
        status = {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "ollama": False,
            "ollama_models": []
        }
        
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                res = json.loads(response.read().decode())
                status["ollama"] = True
                status["ollama_models"] = [m["name"] for m in res.get("models", [])]
        except Exception:
            status["ollama"] = False

        if self.provider == "gemini" or (self.provider == "auto" and status["gemini"]):
            active_name = f"Google Gemini ({self.model_name or 'gemini-1.5-flash'})"
            engine_type = "Cloud API"
        elif self.provider == "openai" or (self.provider == "auto" and status["openai"]):
            active_name = f"OpenAI ({self.model_name or 'gpt-4o-mini'})"
            engine_type = "Cloud API"
        elif self.provider == "groq" or (self.provider == "auto" and status["groq"]):
            active_name = f"Groq Cloud ({self.model_name or 'llama-3.1-8b-instant'})"
            engine_type = "Cloud API"
        elif self.provider == "ollama" or (self.provider == "auto" and status["ollama"]):
            installed_m = status['ollama_models'][0] if status['ollama_models'] else 'tinyllama'
            active_name = f"Local Ollama ({self.model_name or installed_m})"
            engine_type = "Local Offline"
        else:
            active_name = "Deterministic Rule-Based Intelligence Engine"
            engine_type = "Local Built-in (Deterministic)"

        return {
            "active_model": active_name,
            "engine_type": engine_type,
            "available_providers": status
        }

    def generate_narrative_brief(self, context_payload: Dict[str, Any]) -> Dict[str, str]:
        prompt = self._build_prompt(context_payload)
        status = self.get_active_provider_status()["available_providers"]

        # 1. Local Ollama Execution
        if (self.provider in ["ollama", "auto"]) and status["ollama"]:
            try:
                # Use the exact installed model tag (e.g. tinyllama:latest)
                installed_models = status["ollama_models"]
                target_model = self.model_name or (installed_models[0] if installed_models else "tinyllama")

                req_body = json.dumps({
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 250
                    }
                }).encode("utf-8")

                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=req_body,
                    headers={"Content-Type": "application/json"}
                )

                # Set timeout to 25 seconds for CPU generation
                with urllib.request.urlopen(req, timeout=25.0) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    generated_text = res_data.get("response", "").strip()
                    if generated_text:
                        return {
                            "source": f"Local Ollama ({target_model})",
                            "content": generated_text
                        }
            except Exception as e:
                print(f"[ProjectIQ AICopilot] Ollama generation failed: {e}. Falling back to deterministic engine.")

        # 2. Google Gemini API
        gemini_key = os.getenv("GEMINI_API_KEY")
        if (self.provider in ["gemini", "auto"]) and gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(self.model_name or "gemini-1.5-flash")
                response = model.generate_content(prompt)
                return {
                    "source": f"Google Gemini ({self.model_name or 'gemini-1.5-flash'})",
                    "content": response.text.strip()
                }
            except Exception as e:
                print(f"[ProjectIQ AICopilot] Gemini generation failed: {e}")

        # 3. OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if (self.provider in ["openai", "auto"]) and openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are ProjectIQ AI, an executive project risk consultant. Synthesize data into a concise brief without hallucinating numbers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=350
                )
                return {
                    "source": f"OpenAI ({self.model_name or 'gpt-4o-mini'})",
                    "content": resp.choices[0].message.content.strip()
                }
            except Exception as e:
                print(f"[ProjectIQ AICopilot] OpenAI generation failed: {e}")

        # 4. Deterministic Fallback Engine
        return {
            "source": "ProjectIQ Deterministic Intelligence Engine",
            "content": self._build_deterministic_narrative(context_payload)
        }

    def _build_prompt(self, p: Dict[str, Any]) -> str:
        return f"""You are ProjectIQ AI, a project risk advisor. Provide a concise summary based ONLY on this data:
- Project: {p.get('name')}
- Success Probability: {p.get('success_probability'):.1f}% (Risk: {p.get('risk_tier')})
- Goal: ${p.get('goal_usd'):,.0f} (Category Median: ${p.get('benchmark_median_goal'):,.0f})
- Duration: {p.get('duration_days')} days | Prep: {p.get('prep_days')} days
- Key Risk Drivers: {', '.join([d['feature'] for d in p.get('top_risk_drivers', [])[:3]])}

Format output with these headers:
### Executive Summary
### Key Vulnerabilities
### Recommended Interventions
Keep it under 150 words."""

    def _build_deterministic_narrative(self, p: Dict[str, Any]) -> str:
        prob = p.get('success_probability', 50.0)
        tier = p.get('risk_tier', 'MEDIUM')
        goal = p.get('goal_usd', 10000)
        
        narrative = "### Executive Summary\n"
        narrative += f"The project demonstrates an estimated **{prob:.1f}% success probability**, categorizing it as **{tier} RISK**. "
        if tier == "HIGH":
            narrative += f"The primary vulnerability is an elevated funding goal (${goal:,.0f}) relative to pre-launch preparation time."
        elif tier == "LOW":
            narrative += "The project aligns strongly with top-performing historical benchmarks."
        else:
            narrative += "The project exhibits moderate viability with specific optimization opportunities."

        narrative += "\n\n### Strategic Interventions\n"
        for idx, rsk in enumerate(p.get("risk_register", [])[:3], 1):
            narrative += f"{idx}. **{rsk['risk_title']}**: {rsk['suggested_response']}\n"
            
        narrative += "\n*Note: Generated via deterministic decision-support logic.*"
        return narrative