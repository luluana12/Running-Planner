from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Set

import io

import pandas as pd
import streamlit as st

from planner.history import read_history
from planner.prediction import predict_performance
from planner.plan import PlanConfig, build_plan
from planner.utils import format_pace, parse_pace_str

st.set_page_config(page_title="Running Planner", layout="wide")

st.title("Running Planner")

st.sidebar.header("Inputs")

# Upload history
uploaded = st.sidebar.file_uploader("Upload history CSV", type=["csv"])
history_df = None
if uploaded is not None:
	history_df = pd.read_csv(uploaded)
	# normalize types
	if "date" in history_df.columns:
		history_df["date"] = pd.to_datetime(history_df["date"]).dt.strftime("%Y-%m-%d")

# Goal race inputs
race_day = st.sidebar.date_input("Race day", value=date.today() + timedelta(days=56))

distance_label = st.sidebar.selectbox("Race distance", ["5K", "10K", "Half", "Marathon", "Custom (km)"])
custom_km = 0.0
if distance_label == "5K":
	race_km = 5.0
elif distance_label == "10K":
	race_km = 10.0
elif distance_label == "Half":
	race_km = 21.097
elif distance_label == "Marathon":
	race_km = 42.195
else:
	custom_km = st.sidebar.number_input("Custom distance (km)", min_value=1.0, value=10.0, step=0.5)
	race_km = custom_km

goal_pace_str = st.sidebar.text_input("Goal pace (min/km, e.g., 5:00)", value="5:00")
try:
	goal_pace = parse_pace_str(goal_pace_str)
except Exception:
	st.sidebar.error("Invalid goal pace format")
	goal_pace = 5.0

# Availability and preferences
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
avail = st.sidebar.multiselect("Availability days", options=weekdays, default=["Mon", "Tue", "Thu", "Sat", "Sun"]) 
long_run_day_label = st.sidebar.radio("Long run day", options=["Sat", "Sun"], index=0)

strength_yes = st.sidebar.checkbox("Include strength days?", value=True)
strength_days = st.sidebar.slider("Strength days per week", 0, 3, 2) if strength_yes else 0

running_days = st.sidebar.slider("Running days per week", 3, 7, 5)

# Vacations and B-races
vacations = st.sidebar.text_area("Vacation days (YYYY-MM-DD, comma-separated)", placeholder="2025-10-15, 2025-10-16")

b_races_text = st.sidebar.text_area(
	"B-races (YYYY-MM-DD distance_km; one per line)",
	placeholder="2025-10-26 10\n2025-11-16 5",
)

# Compute prediction
if history_df is not None:
	try:
		parsed_df = read_history(uploaded)
	except Exception:
		# If read_history expects path, fallback to dataframe already read
		parsed_df = history_df.copy()
		parsed_df["pace_min_per_km"] = parsed_df.apply(lambda r: r["duration_min"] / max(r["distance_km"], 1e-9), axis=1)
else:
	parsed_df = None

col1, col2 = st.columns(2)

if parsed_df is not None and len(parsed_df) > 0:
	pred = predict_performance(parsed_df, race_km, goal_pace)
	with col1:
		st.subheader("Performance Prediction")
		st.metric("Predicted race pace", format_pace(pred.predicted_pace_min_per_km))
		st.metric("Predicted finish time", f"{pred.predicted_time_min/60:.2f} h")
else:
	with col1:
		st.info("Upload history to see prediction")

# Build plan
avail_idx: Set[int] = {weekdays.index(w) for w in avail}
long_idx = weekdays.index(long_run_day_label)

# Parse vacations
vacation_days: Set[date] = set()
for token in [t.strip() for t in vacations.split(",") if t.strip()]:
	try:
		vacation_days.add(datetime.strptime(token, "%Y-%m-%d").date())
	except Exception:
		st.warning(f"Skipping invalid vacation date: {token}")

# Parse B-races
b_races: Dict[date, float] = {}
for line in [l.strip() for l in b_races_text.splitlines() if l.strip()]:
	try:
		parts = line.split()
		dt = datetime.strptime(parts[0], "%Y-%m-%d").date()
		dist = float(parts[1])
		b_races[dt] = dist
	except Exception:
		st.warning(f"Skipping invalid B-race entry: {line}")

base_pace = goal_pace + 0.3 if parsed_df is None else float(parsed_df["pace_min_per_km"].median())

config = PlanConfig(
	race_day=race_day,
	availability_weekdays=avail_idx,
	long_run_weekday=long_idx,
	strength_days_per_week=strength_days,
	running_days_per_week=running_days,
	vacation_days=vacation_days,
	b_races=b_races,
	goal_pace_min_per_km=goal_pace,
	base_pace_min_per_km=base_pace,
	start_day=date.today(),
)

plan_df = build_plan(config)
with col2:
	st.subheader("Training Plan")
	st.dataframe(plan_df, use_container_width=True)
	csv_buf = io.StringIO()
	plan_df.to_csv(csv_buf, index=False)
	st.download_button("Download Plan CSV", csv_buf.getvalue(), file_name="training_plan.csv", mime="text/csv")

st.caption("Rules: Easy after hard days; quality after easy; progressive long runs; taper before race.")
