from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from .utils import pace_min_per_km, to_date


@dataclass
class HistoryStats:
	mean_pace: float
	weekly_km: float
	fitness_index: float


def read_history(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	assert {"date", "distance_km", "duration_min"}.issubset(df.columns), (
		"CSV must include columns: date, distance_km, duration_min"
	)
	df["date"] = pd.to_datetime(df["date"]).dt.date
	df["pace_min_per_km"] = df.apply(
		lambda r: pace_min_per_km(r["duration_min"], r["distance_km"]), axis=1
	)
	# Normalize type
	if "type" not in df.columns:
		df["type"] = ""
	else:
		df["type"] = df["type"].fillna("").str.lower()

	# Intensity classification: easy / moderate / hard
	# Rule of thumb by pace vs rolling mean and workout type hints
	rolling = (
		df.sort_values("date")["pace_min_per_km"].rolling(window=14, min_periods=1).median()
	)
	threshold_fast = rolling * 0.92
	threshold_slow = rolling * 1.08
	conditions = []
	for i, r in df.iterrows():
		pace = r["pace_min_per_km"]
		type_hint = r.get("type", "")
		if type_hint in {"intervals", "tempo", "track", "race"} or pace < threshold_fast.iloc[i]:
			label = "hard"
		elif type_hint in {"long"} or pace > threshold_slow.iloc[i]:
			label = "easy"
		else:
			label = "moderate"
		conditions.append(label)
	df["intensity"] = conditions

	return df


def summarize_history(df: pd.DataFrame, as_of: Optional[date] = None) -> HistoryStats:
	if as_of is not None:
		df = df[df["date"] <= as_of]
	if len(df) == 0:
		return HistoryStats(mean_pace=0.0, weekly_km=0.0, fitness_index=0.0)

	mean_pace = float(df["pace_min_per_km"].median())
	# Weekly km from last 28 days scaled to weekly
	last_28 = df[df["date"] >= (max(df["date"]) - pd.Timedelta(days=28)).date()]
	total_28 = float(last_28["distance_km"].sum())
	weekly_km = total_28 / 4.0

	# Fitness index: blend of volume and speed (lower pace is better)
	# Scale pace so that lower is higher score
	pace_score = 6.0 / max(mean_pace, 1e-6)
	volume_score = weekly_km / 40.0
	fitness_index = 0.6 * pace_score + 0.4 * volume_score

	return HistoryStats(mean_pace=mean_pace, weekly_km=weekly_km, fitness_index=fitness_index)
