from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional


def to_date(value: str | date | datetime) -> date:
	if isinstance(value, date) and not isinstance(value, datetime):
		return value
	if isinstance(value, datetime):
		return value.date()
	return datetime.strptime(str(value), "%Y-%m-%d").date()


def daterange(start: date, end: date) -> List[date]:
	# inclusive start, inclusive end
	days = (end - start).days
	return [start + timedelta(days=i) for i in range(days + 1)]


def pace_min_per_km(duration_min: float, distance_km: float) -> float:
	if distance_km <= 0:
		return 0.0
	return duration_min / max(distance_km, 1e-9)


def format_pace(min_per_km: float) -> str:
	if min_per_km <= 0:
		return "-"
	minutes = int(min_per_km)
	seconds = int(round((min_per_km - minutes) * 60))
	if seconds == 60:
		minutes += 1
		seconds = 0
	return f"{minutes}:{seconds:02d}/km"


def parse_pace_str(text: str) -> float:
	# Accept forms like "5:00", "5:00/km", or "5.5" (min/km)
	clean = text.strip().lower().replace("/km", "")
	if ":" in clean:
		m, s = clean.split(":", 1)
		return int(m) + int(s) / 60.0
	return float(clean)
