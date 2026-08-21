"""
ProjectIQ V2.0 — Instance-Level SHAP Explainability Engine
Generates local feature attributions, waterfall plots, and risk-vs-success attribution pie charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

class ProjectIQExplainer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.preprocessor = predictor.preprocessor
        self.classifier = predictor.classifier
        
        # Extract transformed column names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out([
            "category_clean", "country_clean", "launch_month", "launch_day_of_week"
        ])
        num_names = [
            "log_goal_usd", "campaign_duration_days", "prep_duration_days", "launch_hour",
            "name_len", "name_word_count", "blurb_len", "blurb_word_count", "has_video",
            "staff_pick", "prelaunch_activated"
        ]
        self.transformed_feature_names = list(num_names) + list(cat_names)
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_instance(self, input_df: pd.DataFrame) -> dict:
        """
        Computes SHAP values for a single input vector.
        """
        X_trans = self.preprocessor.transform(input_df)
        shap_vals = self.explainer(X_trans)
        
        # Single row explanation
        row_shap = shap_vals[0]
        row_shap.feature_names = self.transformed_feature_names
        
        # Rank top drivers
        values = row_shap.values
        sorted_indices = np.argsort(np.abs(values))[::-1]
        
        top_drivers = []
        pos_impact_sum = 0.0
        neg_impact_sum = 0.0

        for val in values:
            if val > 0:
                pos_impact_sum += float(val)
            else:
                neg_impact_sum += abs(float(val))

        for idx in sorted_indices[:8]:
            fname = self.transformed_feature_names[idx]
            val = float(values[idx])
            sentiment = "Supports Success" if val > 0 else "Increases Risk"
            top_drivers.append({
                "feature": fname,
                "shap_value": val,
                "impact": abs(val),
                "direction": sentiment
            })
            
        return {
            "shap_explanation": row_shap,
            "top_drivers": top_drivers,
            "pos_impact_sum": pos_impact_sum,
            "neg_impact_sum": neg_impact_sum,
            "base_value": float(self.explainer.expected_value)
        }

    def generate_waterfall_plot(self, row_shap, output_path: str = "instance_shap_waterfall.png"):
        """
        Renders and saves an instance-level SHAP waterfall figure.
        """
        fig, ax = plt.subplots(figsize=(9, 5.5))
        shap.plots.waterfall(row_shap, max_display=10, show=False)
        plt.title("ProjectIQ: Local Risk/Success Factor Attribution (SHAP)", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path

    def generate_pie_chart(self, pos_sum: float, neg_sum: float, output_path: str = "instance_shap_pie.png"):
        """
        Renders a donut/pie chart displaying the net balance of risk-increasing vs success-supporting forces.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Handle zero or negligible values gracefully
        if pos_sum + neg_sum == 0:
            sizes = [50, 50]
        else:
            sizes = [neg_sum, pos_sum]

        labels = ['Risk-Increasing Forces', 'Success-Supporting Forces']
        colors = ['#EF4444', '#10B981'] # Red & Green

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops=dict(color="#1E293B", fontsize=10, weight="bold"),
            wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)

        plt.title("Net Factor Attribution Balance", fontsize=12, weight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path