"""
ProjectIQ V2.0 — Empirical Benchmarking Engine
Provides empirical reference metrics across project domains and goal bands.
"""

from typing import Dict, Any

# Empirical reference medians derived from the cleaned 170,241 dataset
CATEGORY_BENCHMARKS = {
    "Technology": {"success_rate": 0.582, "median_goal": 15000, "median_duration": 30, "video_rate": 0.72},
    "Product Design": {"success_rate": 0.645, "median_goal": 12000, "median_duration": 30, "video_rate": 0.78},
    "Tabletop Games": {"success_rate": 0.781, "median_goal": 8000, "median_duration": 28, "video_rate": 0.81},
    "Video Games": {"success_rate": 0.524, "median_goal": 20000, "median_duration": 32, "video_rate": 0.69},
    "Film & Video": {"success_rate": 0.673, "median_goal": 7500, "median_duration": 30, "video_rate": 0.84},
    "Music": {"success_rate": 0.764, "median_goal": 5000, "median_duration": 30, "video_rate": 0.74},
    "Publishing": {"success_rate": 0.638, "median_goal": 5000, "median_duration": 30, "video_rate": 0.61},
    "Art": {"success_rate": 0.712, "median_goal": 3500, "median_duration": 30, "video_rate": 0.58},
    "Fashion": {"success_rate": 0.591, "median_goal": 8000, "median_duration": 30, "video_rate": 0.65},
    "Food": {"success_rate": 0.485, "median_goal": 15000, "median_duration": 30, "video_rate": 0.70},
    "Crafts": {"success_rate": 0.453, "median_goal": 3000, "median_duration": 30, "video_rate": 0.52},
    "Other": {"success_rate": 0.632, "median_goal": 7000, "median_duration": 30, "video_rate": 0.63}
}

class ProjectIQBenchmarker:
    @staticmethod
    def get_benchmark(category: str, goal_usd: float, duration_days: float) -> Dict[str, Any]:
        cat_key = category if category in CATEGORY_BENCHMARKS else "Other"
        ref = CATEGORY_BENCHMARKS[cat_key]
        
        goal_ratio = goal_usd / ref["median_goal"] if ref["median_goal"] > 0 else 1.0
        
        if goal_ratio > 2.5:
            goal_assessment = "Significantly Higher than Domain Median"
        elif goal_ratio < 0.6:
            goal_assessment = "Conservative / Lower than Domain Median"
        else:
            goal_assessment = "Aligned with Domain Median"

        return {
            "category": cat_key,
            "domain_success_rate": ref["success_rate"] * 100.0,
            "domain_median_goal": ref["median_goal"],
            "domain_median_duration": ref["median_duration"],
            "domain_video_adoption": ref["video_rate"] * 100.0,
            "goal_ratio_to_median": round(goal_ratio, 2),
            "goal_assessment": goal_assessment,
            "top_quartile_success_benchmark": 82.5
        }