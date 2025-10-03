from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

import math
import pandas as pd

from .utils import daterange, format_pace


@dataclass
class PlanConfig:
	race_day: date
	availability_weekdays: Set[int]  # 0=Mon ... 6=Sun
	long_run_weekday: int  # 5=Sat or 6=Sun typically
	strength_days_per_week: int
	running_days_per_week: int
	vacation_days: Set[date]
	b_races: Dict[date, float]  # date -> distance_km
	goal_pace_min_per_km: float
	base_pace_min_per_km: float
	start_day: date


def weekday_name(idx: int) -> str:
	return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][idx]


def build_plan(config: PlanConfig) -> pd.DataFrame:
	# Plan from next Monday (or today) until race_day
	all_days = daterange(config.start_day, config.race_day)

	schedule: List[dict] = []

	# Determine weekly pattern of run days based on availability
	# Greedy: pick long run day + distribute remaining run days across available days
	for d in all_days:
		entry = {
			"date": d,
			"weekday": weekday_name(d.weekday()),
			"type": "rest",
			"distance_km": 0.0,
			"target_pace": "-",
			"notes": "",
			"strength": False,
		}
		schedule.append(entry)

	# Mark vacations as rest locked
	vacation_set = set(config.vacation_days)

	# Identify weeks and assign structure
	by_week: Dict[Tuple[int, int], List[int]] = {}
	for i, e in enumerate(schedule):
		iso = e["date"].isocalendar()
		key = (iso.year, iso.week)
		by_week.setdefault(key, []).append(i)

	weekly_long_km = 12.0  # starting long run distance
	if config.running_days_per_week >= 5:
		weekly_long_km = 14.0

	for week_key, idxs in by_week.items():
		week_dates = [schedule[i]["date"] for i in idxs]
		available_in_week = [i for i in idxs if schedule[i]["date"].weekday() in config.availability_weekdays and schedule[i]["date"] not in vacation_set]

		# Choose run days
		run_slots = set()
		# Ensure long run day if available
		long_idx = None
		for i in available_in_week:
			if schedule[i]["date"].weekday() == config.long_run_weekday:
				long_idx = i
				break
		if long_idx is not None:
			run_slots.add(long_idx)

		# Fill remaining run days
		remaining = config.running_days_per_week - len(run_slots)
		for i in available_in_week:
			if remaining <= 0:
				break
			if i in run_slots:
				continue
			run_slots.add(i)
			remaining -= 1

		# Assign workouts following rules
		# 1 long, 1-2 quality depending on days/week, others easy or recovery
		quality_count = 2 if config.running_days_per_week >= 5 else 1

		# Place B-races as quality
		b_race_slots: Set[int] = set()
		for i in idxs:
			d = schedule[i]["date"]
			if d in config.b_races:
				b_race_slots.add(i)
				run_slots.add(i)

		# Set workouts
		for i in idxs:
			if i not in run_slots:
				continue
			d = schedule[i]["date"]
			is_long = d.weekday() == config.long_run_weekday
			if d in vacation_set:
				continue

			if i in b_race_slots:
				# B-race
				dist = config.b_races[d]
				schedule[i]["type"] = "b-race"
				schedule[i]["distance_km"] = dist
				schedule[i]["target_pace"] = "race effort"
				schedule[i]["notes"] = "Tune-up race"
				continue

			if is_long:
				# Long run progressive build
				schedule[i]["type"] = "long"
				schedule[i]["distance_km"] = round(weekly_long_km, 1)
				schedule[i]["target_pace"] = f"{format_pace(config.base_pace_min_per_km + 0.5)}"
				continue

		# Assign quality days away from long run and not consecutive hard days
		# First pass: mark all as easy
		for i in idxs:
			if i in run_slots and schedule[i]["type"] == "rest":
				schedule[i]["type"] = "easy"
				schedule[i]["distance_km"] = 6.0 if config.running_days_per_week <= 4 else 8.0
				schedule[i]["target_pace"] = f"{format_pace(config.base_pace_min_per_km + 0.3)}"

		# Promote some to quality respecting spacing
		quality_chosen = 0
		for i in idxs:
			if quality_chosen >= quality_count:
				break
			if schedule[i]["type"] != "easy":
				continue
			# avoid adjacent to long
			prev_i = i - 1 if i - 1 in idxs else None
			next_i = i + 1 if i + 1 in idxs else None
			adjacent_long = False
			for j in [prev_i, next_i]:
				if j is None:
					continue
				if schedule[j]["type"] == "long":
					adjacent_long = True
					break
			if adjacent_long:
				continue
			schedule[i]["type"] = "quality"
			schedule[i]["notes"] = "Intervals/tempo based on goal pace"
			schedule[i]["distance_km"] = 8.0 if config.running_days_per_week <= 4 else 10.0
			schedule[i]["target_pace"] = f"{format_pace(config.goal_pace_min_per_km)}"
			quality_chosen += 1

		# Recovery after hard days: convert the day after quality/long to recovery or rest if not already set
		for i in idxs:
			if schedule[i]["type"] in {"quality", "long", "b-race"}:
				next_idx = i + 1 if i + 1 in idxs else None
				if next_idx is not None and schedule[next_idx]["type"] in {"easy", "rest"}:
					schedule[next_idx]["type"] = "recovery" if schedule[next_idx]["type"] == "easy" else "rest"
					schedule[next_idx]["distance_km"] = 5.0 if schedule[next_idx]["type"] == "recovery" else 0.0
					schedule[next_idx]["target_pace"] = f"{format_pace(config.base_pace_min_per_km + 0.5)}" if schedule[next_idx]["type"] == "recovery" else "-"

		# Slightly increase long run most weeks, cut back every 4th week
		weekly_long_km = weekly_long_km * 1.05
		if weekly_long_km > 28:
			weekly_long_km = 28
		week_number = week_key[1]
		if week_number % 4 == 0:
			weekly_long_km = max(weekly_long_km * 0.85, 16)

	# Strength days: fill non-run, available days
	for week_key, idxs in by_week.items():
		strength_to_assign = config.strength_days_per_week
		for i in idxs:
			if strength_to_assign <= 0:
				break
			if schedule[i]["date"] in vacation_set:
				continue
			if schedule[i]["date"].weekday() not in config.availability_weekdays:
				continue
			if schedule[i]["type"] == "rest":
				schedule[i]["strength"] = True
				schedule[i]["notes"] = (schedule[i]["notes"] + "; " if schedule[i]["notes"] else "") + "Strength"
				strength_to_assign -= 1

	# Taper last 2 weeks
	for e in schedule:
		days_to_race = (config.race_day - e["date"]).days
		if days_to_race <= 14:
			if e["type"] == "long":
				fraction = 0.7 if days_to_race > 7 else 0.5
				e["distance_km"] = round(e["distance_km"] * fraction, 1)
			elif e["type"] == "quality":
				e["distance_km"] = round(e["distance_km"] * 0.7, 1)
				e["notes"] = "Sharpening"

	# Race day entry
	for i, e in enumerate(schedule):
		if e["date"] == config.race_day:
			e["type"] = "race"
			e["distance_km"] = 5.0
			e["target_pace"] = "race"
			e["notes"] = "Goal race"

	df = pd.DataFrame(schedule)
	return df
