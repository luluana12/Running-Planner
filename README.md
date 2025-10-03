# Running Planner

A Streamlit app that predicts race performance from training history and generates a personalized training plan with constraints (availability, vacations, B-races, long-run day, strength days, running days/week).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## History CSV format

Provide a CSV with columns:
- date: YYYY-MM-DD
- distance_km: float (km)
- duration_min: float (minutes)
- type: optional string (e.g., easy, tempo, intervals, long)

See `sample_data/sample_history.csv`.

## Features
- Performance prediction from recent training (rolling 6-week fitness index)
- Plan rules: easy-after-hard, hard-after-easy, long run on preferred day
- Handles vacations, availability, B-races, strength days, running days/week
- Export plan to CSV
