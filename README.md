# ProjectIQ: AI-Driven Project Success & Risk Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-orange)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/XAI-TreeSHAP-brightgreen)](https://shap.readthedocs.io/)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ProjectIQ is an explainable machine learning decision-support platform designed to evaluate project success probability, quantify risk exposure, and deliver actionable managerial interventions at pre-launch ($T_0$). 

Trained on **170,241 historical crowdfunding initiatives**, ProjectIQ shifts predictive modeling from static scoring into an interactive decision-intelligence loop: **Predict → Explain → Simulate → Benchmark → Advise → Report**.

---

## System Architecture
USER / DECISION MAKER
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │    Project Parameters at Launch (T₀)  │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │ Preprocessing & Feature Transformation│
                  │ (Log-Goal, Timing, Text Signals, OHE) │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │    Fitted XGBoost Classifier Pipeline │
                  │    (Holdout ROC-AUC: 0.8646, N=34,049)│
                  └───────────────────┬───────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
    ┌─────────────────────────┐               ┌─────────────────────────┐
    │ Calibrated Risk Scoring │               │ Instance-Level TreeSHAP │
    │ (1 - P(Success) Gauge)  │               │ (Waterfall & Pie Share) │
    └────────────┬────────────┘               └────────────┬────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │      Decision Intelligence Core       │
                  ├───────────────────────────────────────┤
                  │ • What-If Sensitivity Simulator       │
                  │ • Domain Empirical Benchmarking       │
                  │ • Structured Project Risk Register    │
                  └───────────────────┬───────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
    ┌─────────────────────────┐               ┌─────────────────────────┐
    │ Resilient AI Copilot    │               │ Programmatic PDF Engine │
    │ (Gemini/Ollama/Fallback)│               │ (ReportLab 2-Page Brief)│
    └─────────────────────────┘               └─────────────────────────┘
	
---

## Version Evolution: V1.0 vs V2.0

| Architectural Layer | V1.0 Baseline (Dissertation MVP) | V2.0 Decision Intelligence Platform |
| :--- | :--- | :--- |
| **Prediction Focus** | Binary Success / Failure Classification | Calibrated Probability & 3-Tier Risk Score ($T_0$) |
| **Model Explainability** | Static Global SHAP Beeswarm Chart | Instance-Level Dynamic TreeSHAP Waterfall & Pie Share |
| **Decision Support** | Static Heuristic Rules | Non-Causal What-If Scenario Sensitivity Simulator |
| **Domain Context** | None | Empirical Category Benchmarking (Medians & Quantiles) |
| **Risk Architecture** | Ad-hoc text outputs | Structured Risk Register (Evidence vs Action split) |
| **Executive Narrative** | Fixed string templates | Resilient Multi-Provider AI Copilot (Gemini / Ollama) |
| **Exportable Assets** | None | Programmatic 2-Page PDF Decision Brief (`ReportLab`) |
| **Code Structure** | Monolithic single-file script | Modular clean-package architecture (`src/` modules) |

---

## Technical Validation & Benchmarking

All models evaluated strictly on a 20% unseen test holdout ($N = 34,049$ projects) under a **Zero-Target-Leakage Firewall** ($T_0$ boundary):

| Model Candidate | ROC-AUC | Accuracy | Precision | Recall | F1-Score | Brier Calibration Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.8646** | **79.42%** | **0.8077** | **0.8853** | **0.8447** | **0.1411** |
| **Random Forest** | 0.8525 | 78.29% | 0.8002 | 0.8719 | 0.8345 | 0.1544 |
| **Logistic Regression** | 0.8376 | 77.47% | 0.7938 | 0.8648 | 0.8278 | 0.1546 |

---

## Directory Structure

```text
ProjectIQ/
│
├── app.py                          # Streamlit UI controller
├── pipeline.py                     # Training pipeline & leakage firewall
├── generate_shap.py                # Global SHAP attribution builder
├── requirements.txt                # System dependencies
├── README.md                       # Comprehensive system documentation
├── projectiq_model.joblib          # Serialized production pipeline
├── model_features.joblib           # Pre-launch feature schema contract
│
├── src/                            # Modular Decision Intelligence Core
│   ├── __init__.py
│   ├── prediction.py               # T₀ inference & calibrated risk scoring
│   ├── explainability.py          # Dynamic SHAP waterfall & pie chart generation
│   ├── scenarios.py                # Controllable What-If sensitivity simulator
│   ├── benchmarking.py            # Historical domain medians & quartile lookups
│   ├── recommendations.py         # Structured risk register generator
│   ├── llm.py                      # Multi-provider LLM Copilot (Gemini/Ollama/Offline)
│   └── reporting.py                # Automated ReportLab PDF brief generator
│
└── docs/                           # Academic & Research Assets
    ├── Project Charter.pdf         # Problem framing & scope
    └── Dissertation_Guide.docx     # Academic structural guidelines
