from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .history import summarize_history


@dataclass
class PerformancePrediction:
	predicted_pace_min_per_km: float
	predicted_time_min: float


def predict_performance(
	history_df: pd.DataFrame,
	distance_km: float,
	goal_pace_min_per_km: Optional[float] = None,
) -> PerformancePrediction:
	stats = summarize_history(history_df)
	if stats.mean_pace <= 0 or distance_km <= 0:
		return PerformancePrediction(0.0, 0.0)

	# Use recent fitness: faster than median pace if fitness_index high
	# Map fitness_index ~ [0.5, 2.0] to pace multiplier ~ [1.05, 0.90]
	fitness = max(0.5, min(stats.fitness_index, 2.0))
	mult = np.interp(fitness, [0.5, 2.0], [1.05, 0.90])
	pred_pace = stats.mean_pace * mult

	# If user provided goal pace, bias towards it slightly
	if goal_pace_min_per_km and goal_pace_min_per_km > 0:
		pred_pace = 0.7 * pred_pace + 0.3 * goal_pace_min_per_km

	pred_time_min = pred_pace * distance_km
	return PerformancePrediction(predicted_pace_min_per_km=float(pred_pace), predicted_time_min=float(pred_time_min))
