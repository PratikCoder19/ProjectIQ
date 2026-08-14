# ProjectIQ: AI-Driven Project Success & Risk Decision Support System

> **Academic Specialization:** PGDM in Data Science  
> **Evaluation Framework:** Academic Dissertation & Decision-Support MVP  
> **Prediction Milestone:** $T_0$ (Strict Pre-Launch Decision Point — Zero Target Leakage)  
> **Status:** Working Proof of Concept (v0.1)

---

## Executive Summary (For Non-Technical Evaluators & Faculty)

Project managers, product leaders, and startup entrepreneurs frequently launch campaigns and capital-intensive initiatives under extreme uncertainty. Historically, **over 36% of crowdfunding and innovation projects fail**, leading to wasted capital, missed market windows, and damaged organizational reputation.

**ProjectIQ** is an explainable Artificial Intelligence (AI) and Machine Learning (ML) Decision Support System. By analyzing **170,241 historical projects**, ProjectIQ provides three core capabilities at the exact moment of project inception ($T_0$):
1. **Calibrated Outcome Estimation:** Predicts the probability of project success ($79.42\%$ accuracy, $0.8646$ ROC-AUC).
2. **Explainable Risk Profiling:** Uses SHAP (game-theoretic AI explainability) to isolate the top drivers of risk (e.g., funding goal size, schedule duration, pre-launch preparation).
3. **Prescriptive Action Recommendations:** Converts raw predictive scores into concrete managerial interventions (e.g., milestone staging, duration compression, pre-launch waitlists) before a single dollar is committed.

---

## 🎯 The Research Problem & Core Question

### Primary Research Question
> *"Can publicly available project data and machine-learning techniques be used to develop an explainable decision-support framework for assessing project success/risk and identifying actionable risk factors?"*

### Why $T_0$ Pre-Launch Prediction Matters (Academic Rigor & Leakage Prevention)
In data science, **Target Leakage** occurs when a model is trained using data that would not realistically be available at the time of decision-making. 

Many naive crowdfunding prediction models achieve artificial $99\%$ accuracy by using post-launch variables like *number of backers*, *funds pledged to date*, or *staff curation after hitting targets*. 

**ProjectIQ strictly enforces a zero-leakage boundary:**
* **Excluded Variables (Post-Launch Contaminants):** `pledged`, `backers_count`, `percent_funded`, `usd_pledged`, `spotlight`, `state_changed_at`.
* **Included Variables (Legitimate Pre-Launch Decisions):** Target budget goal, planned campaign duration, preparation lead time, launch timing (day, hour, month), project title/description metrics, and video pitch availability.

---

## 📊 Dataset Overview & Health Summary

The research utilizes a consolidated public dataset of Kickstarter project campaigns:

| Metric / Dimension | Raw Data Property | Cleaned Analytical Scope |
| :--- | :--- | :--- |
| **Total Observations** | 191,384 campaigns | **170,241 resolved campaigns** |
| **Temporal Span** | 2009 to 2026 | **2009 to 2026** (Epoch artifacts removed) |
| **Target Distribution** | 5 raw states | **63.22% Successful ($1$) vs. 36.78% Failed ($0$)** |
| **Currency Standardization** | Multiple Global Currencies | Standardized to **Normalized USD** via static exchange rates |
| **Partitioning Strategy** | Stratified Holdout | **80% Training ($136,192$) / 20% Testing ($34,049$)** |

---

## ⚙️ Model Benchmarking & Empirical Results

Three distinct machine learning paradigms were trained and evaluated against the held-out test set ($34,049$ unseen projects):

| Model Algorithm | Paradigm Description | ROC-AUC | Accuracy | F1-Score | Brier Score (Calibration) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Interpretable Linear Baseline | `0.8376` | `77.47%` | `0.8283` | `0.1546` |
| **Random Forest** | Non-Linear Ensemble (Bagging) | `0.8525` | `78.29%` | `0.8418` | `0.1544` |
| **XGBoost (Champion)** | Gradient Boosted Decision Trees | **`0.8646`** | **`79.42%`** | **`0.8447`** | **`0.1411`** |

### Understanding the Evaluation Metrics in Plain English:
* **ROC-AUC ($0.8646$):** Measures the system's ability to rank a successful project higher than a failing project across all decision thresholds. A score above $0.85$ signifies an industrial-grade predictive classifier.
* **Brier Score ($0.1411$):** Measures the precision of calculated probabilities (closer to $0.0$ is perfect). This guarantees that a predicted "$75\%$ success chance" genuinely reflects historical reality.

---

## 🔍 Model Explainability & Key Business Findings (SHAP)

Using **SHAP (SHapley Additive exPlanations)**, ProjectIQ breaks open the "black box" of machine learning to isolate which decisions most directly influence outcome risk:

![SHAP Summary Plot](shap_summary.png)

### Key Actionable Insights for Project Managers:
1. **Funding Goal Calibration (`log_goal_usd` — Impact: 0.758):** The single largest risk factor. Setting a capital goal above platform medians dramatically depresses success likelihood unless partitioned into staged funding milestones.
2. **Platform Promotion (`staff_pick` — Impact: 0.525):** Platform curation yields a substantial upward shift in backer confidence.
3. **The Preparation Window (`prep_duration_days` — Impact: 0.327):** Rushed projects (drafted and launched within $<14$ days) face severe failure penalties. A structured preparation lead time is vital for marketing asset readiness.
4. **Campaign Duration Sweet Spot (`campaign_duration_days` — Impact: 0.302):** Protracted campaigns ($>35$ days) suffer from momentum decay and donor fatigue. The optimal campaign window is **25 to 32 days**.
5. **Pitch Video Inclusion (`has_video` — Impact: 0.280):** Having a video pitch significantly improves conversion probability over text-only campaigns.

---

## 🖥️ System Architecture & Workflow

```text
[ Project Creator Inputs Parameters at T₀ ]
  (Goal, Timeline, Category, Blurb, Video)
                    │
                    ▼
[ Data Preprocessing & Leakage Firewall ]
  (USD Normalization, Log Transform, One-Hot Encoding)
                    │
                    ▼
[ Calibrated Machine Learning Engine (XGBoost) ]
  (Evaluates multidimensional interaction patterns)
                    │
                    ▼
[ Decision Support & Recommendation Engine ]
  ├── 1. Probability of Success & Risk Tier (Low / Medium / High)
  ├── 2. SHAP-Derived Root Cause Risk Identification
  └── 3. Prescriptive Managerial Interventions
```

---

## 🚀 Quick Start Guide: Running the Application Locally

### Prerequisites
* Python 3.9 or higher installed on your system.

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/projectiq-decision-support.git](https://github.com/your-username/projectiq-decision-support.git)
   cd projectiq-decision-support
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Interactive Decision-Support Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *The application will automatically open in your default browser at `http://localhost:8501`.*

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.x
* **Data Ingestion & Processing:** Pandas, NumPy, PyArrow
* **Machine Learning Pipeline:** Scikit-Learn, XGBoost
* **Model Explainability:** SHAP (TreeExplainer)
* **Interactive UI / Decision Dashboard:** Streamlit
* **Model Serialization:** Joblib

---

## 📋 Dissertation Academic Mapping (IIBS Guidelines Compliance)

* **Chapter 1: Introduction** — Theoretical foundation of crowdfunding, project risk management, and the need for early decision support.
* **Chapter 2: Review of Literature & Research Design** — Literature review on project failure drivers, hypothesis framing, and leakage-free methodology design.
* **Chapter 3: Profile of Organization & Ecosystem** — Detailed overview of the Kickstarter platform dynamics, project categories, and backer demographics.
* **Chapter 4: Data Analysis & Interpretation** — Data cleaning, baseline vs. champion modeling, ROC-AUC/Brier metrics, and SHAP explainability.
* **Chapter 5: Summary of Findings, Conclusions & Recommendations** — Strategic takeaways, managerial guidelines for project creators, limitations, and future AI product roadmap.

---

## ⚖️ Limitations & Academic Integrity
* **Secondary Data Scope:** The model is trained on public platform campaign data; it does not observe off-platform private ad spend or external creator social media followings.
* **Prediction Boundary:** Results represent pre-launch risk estimations at $T_0$ and should be paired with ongoing project execution monitoring.
