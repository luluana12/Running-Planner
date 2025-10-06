import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

#Helper
#List of weekdays that will be reused to schedule runs
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#turn HH:MM:SS or MM:SS into minutes, and minutes back to pace strings
#parse_hms_to_minutes("1:30:00") -> 90.0
# parse_hms_to_minutes(x): accept "HH:MM:SS" or "MM:SS" or already-minutes and return minutes (float)
# pace_to_min_per_mile(s): wrapper that reuses the parser (useful for pace strings like "09:00")
# min_to_pace_str(x): convert minutes (float) back into a "MM:SS" string for display
def parse_hms_to_minutes(x):
    #handles missing values (None, Not a number, etc.)
    if pd.isna(x):
        return np.nan

    #Convert to string so we can parse consistently
    text = str(x).strip()
    parts = text.split(':')

    try:
        if len(parts) == 3: #HH:MM:SS
            hours, minutes, seconds = map(float,parts) #[float(p) for p in parts]
            return hours * 60 + minutes + seconds / 60
        elif len(parts) == 2: #MM:SS
            minutes, seconds = map(float,parts)
            return minutes + seconds / 60
        else:
            return float(text) #assume its already in minutes ("75" or 75)
    except Exception:
        return np.nan
    
def pace_to_min_per_mile(s):
    #if pace is missing, return NaN
    if pd.isna(s): 
        return np.nan 
    return parse_hms_to_minutes(s)

def min_to_pace_str(x):
    #if value is missing, return empty string for neatness
    if pd.isna(x):
        return ""
    #whole minutes + remaining seconds
    m = int(x)
    s = int(round((x - m) * 60)) #handle rounding edge case

    if s == 60: 
        m +=1 
        s = 0
        #always show 2 digits for minutes and seconds ( 8 -> 09 )
    return f"{m:02d}:{s:02d}"   

# Data loading and preprocessing
# Load your CSV regardless of column names by letting YOU map them in the sidebar
# We then standardize to these internal columns: Date, Distance, elapsed_time, Pace (optional)
# We compute: distance_miles (always in miles), elapsed_min(minutes), pace_min_per_mile (minutes per mile)
def load_runs(path,date_col, dist_col, time_col,pace_col, unit):
    try:
        df = pd.read_csv(path) #reads the CSV into a dataframe(df)
    except Exception as e:
        st.warning(f"Could not load file: {e}")
        return None
    #rename user-provided column names to our internal lowercase names
    rename = {date_col:'date', dist_col:'distance', time_col:'elapsed_time'}
    if pace_col: 
        rename[pace_col] = 'pace'
    df = df.rename(columns=rename, errors='ignore')
    df.columns = [c.lower() for c in df.columns] #make all columns lowercase

    # Validate minimum columns: we must have date, distance, elapsed_time
    if not {"date", "distance", "elapsed_time"} .issubset(df.columns):
        st.error("Please map the correct column names in the sidebar.")
        return None
    
    #Parse dates, drop rows with invalid dates
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date"])

    # convert distances to miles if needed
    if str(unit).lower().startswith("km"):
        df["distance_miles"] = df["distance"].astype(float) * 0.621371
    else:
        df["distance_miles"] = df["distance"].astype(float)
    #convert elapsed_time to minutes (float)
    df["elapsed_min"] = df["elapsed_time"].apply(parse_hms_to_minutes)

    #Compute pace (min/mile). If a pace column is present, try to parse it; otherwise derive it
    if "pace" in df.columns:
        df["pace_min_per_mile"] = df["pace"].apply(pace_to_min_per_mile)
        needs_calc = df["pace_min_per_mile"].isna()
        df.loc[needs_calc, "pace_min_per_mile"] = df.loc[needs_calc, "elapsed_min"] / df.loc[needs_calc, "distance_miles"]
    else:
        df["pace_min_per_mile"] = df["elapsed_min"] / df["distance_miles"]

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["distance_mi","elapsed_min","pace_min_per_mi"])
        .sort_values("date")
    )
    return df

# Performance estimation
# estimate_next_race(runs, goal_distance_miles, goal_pace_str) 
# uses your last 6 weeks of pace as a "recent fitness" signal
# Blends with your goal pace (70 % recent + 30 % goal) to avoid unrealistic predictions
# Converts suggested pace * distance into a predicted finish time
# Estimate next race performance (history → prediction)
def estimate_next_race(runs,goal_distance_miles, goal_pace_str):
    if runs is None or runs.empty:
        return {"pred_time_str":"", "suggested_pace_str":""}
    
    # Filter to last 6 weeks (42 days)
    recent = runs[runs["date"] >= runs["date"].max() - pd.Timedelta(days=42)]
    recent_pace = (
        recent["pace_min_per_mile"].mean() 
        if not recent.empty 
        else runs["pace_min_per_mile"].mean()
    )
    # Goal pace input (may be missing or invalid). If blank, fall back to recent pace
    goal_pace = pace_to_min_per_mile(goal_pace_str) if goal_pace_str else recent_pace

    # Blend recent pace with goal pace
    suggested = 0.7 * recent_pace + 0.3 * goal_pace # <-- history + goal blend
    total_min = suggested * float(goal_distance_miles) 

    # Minutes -> HH:MM:SS text
    hours = int(total_min // 60) 
    minutes = int(total_min % 60)
    seconds = int(round((total_min - hours * 60 - minutes) * 60))

    return {
        "pred_time_str": f"{hours}:{minutes:02d}:{seconds:02d}",
        "suggested_pace_str": min_to_pace_str(suggested)
    }

# Input Parses for constraints
# parse_dates_list("YYYY-MM-DD,YYYY-MM-DD,...") -> {date, ...}
# parse_b_races("YYYY-MM-DD:distance, ...") -> {date: distance, ...}"
#Parse constraints for vacations & B-races - converts your text inputs into usable dates
def parse_date_list(s):
    out = set()
    if not str(s).strip(): 
        return out
    for t in str(s).split(","):
        t = t.strip()
        try: 
            out.add(pd.to_datetime(t).date())
        except Exception:
            pass
    return out

def parse_b_races(s):
    #"YYYY-MM-DD:distance"
    out = {}
    if not str(s).strip(): 
        return out
    for t in str(s).split(","):
        t = t.strip()
        if ":" in t:
            d, dist = t.split(":", 1)
            try:
                out[pd.to_datetime(d).date()] = float(dist)
            except Exception:
                pass
    return out

# Availability Handler
# Select run days from your availability
#ensures we schedule runs only on days you can run (and always include your long run day)
def pick_run_days(allowed_days, long_run_day, run_days_per_week):
    # Start list with the long-run day
    days = [long_run_day] if long_run_day not in allowed_days else [long_run_day]

    #Fill from allowed days, skipping duplicates and the already added long run day
    for d in DAYS:
        if d == long_run_day: 
            continue
        if d in allowed_days and len(days) < run_days_per_week:
            days.append(d)

        #If still short, fill any remaining weekdays
        for d in DAYS:
            if len(days) == run_days_per_week: 
                break
            if d not in days:
                days.append(d)
        return days[:run_days_per_week]
    
 #Build the training plan
    # Enforces: no back to back hard days, recovery after hard (Quality/Long/B-race)
    # Respects: vacations, B-Races, availability, long run day, strength days
    # Enforces hard -> recovery sequencing , places Long Run, inserts B-races and vacations, and optionally adds strength
def build_plan(
         race_day, 
         goal_distance_miles,
         goal_pace_str,
         run_days_per_week,
         long_run_day,
         allowed_days,
         strength_yes,
         strength_days_per_week,
         vacations_str,
         b_races_str
         ):
    #Plan duration: from today until race day (minimum 4 weeks)
    today = pd.Timestamp.today().normalize().date()
    race_day = pd.to_datetime(race_day).date()
    days_to_race = (race_day - today).days
    weeks = max(4, int(np.ceil(days_to_race / 7))) #at least 4 weeks
    allowed_set = set(allowed_days)

    if long_run_day not in allowed_set:
        st.warning(
            f"Long run day ({long_run_day}) is not in your availability. "
              " I'll still schedule it-adjust if needed."
            )
    run_weekdays = pick_run_days(
        allowed_days=allowed_set, 
        long_run_day=long_run_day, 
        run_days_per_week=run_days_per_week
        )
    vacations = parse_date_list(vacations_str)
    b_races = parse_b_races(b_races_str)

    #Long run progression and weekly mileage targets
    # Start at ~50% of target long run, progress weekly to ~85% of race distance
    target_long = max(6.0,0.85*goal_distance_miles)
    start_long = max(4.0,0.5*target_long)
    inc = max(0.5, (target_long - start_long) / max(1, weeks - 2))

    rows = []

    for w in range(weeks):
        week_start = today + timedelta(days=w*7)
        long_dist = round(min(target_long, start_long + w * inc),1) # Long run distance for the week
        weekly_miles = max(long_dist*1.6, long_dist + 6) #simple (very tough) weekly mileage heuristic
        other_days = max(0,run_days_per_week - 1) 
        base_easy = max(3.0, (weekly_miles - long_dist) / max(1, other_days))

        used_quality = False
        recovery_reserved = False

        for day_index, d_name in enumerate(DAYS):
            day_date = (week_start + timedelta(days=day_index)).date()
            if day_date > race_day:
                continue 
        
        #Default entry is an off day
            entry = {
                "date": day_date,
                "day": d_name,
                "type": "Off", 
                "distance_miles": 0.0, 
                "notes": ""
            }

        #Respect vacations first
            if day_date in vacations:
                entry["notes"] = "Vacation/Rest"
            #B race ovverrides: schedule race + force next day recovery    
            elif day_date in b_races:
                entry["type"] = "B-Race"
                entry["distance_miles"] = b_races[day_date]
                entry["notes"] = "Hard effort; recover tomorrow"
                used_quality = True
                recovery_reserved = True
            #Lomg run on your chosen day (if that day is among your run weekdays)
            elif d_name == long_run_day and d_name in run_weekdays:
                entry["type"] = "Long Run"
                entry["distance_miles"] = long_dist
                entry["notes"] = "Easy, steady pace"
                recovery_reserved = True
            # Fill other running days from your availability
            elif d_name in run_weekdays:
                if recovery_reserved:
                    entry["type"] = "Recovery"
                    entry["distance_miles"] = max(2.0, round(base_easy - 1.0,1))
                    entry["notes"] = "Very easy or off-feel jog"
                    recovery_reserved = False
                elif not used_quality:
                    entry["type"] = "Quality"
                    entry["distance_miles"] = round(max(4.0,base_easy),1)
                    entry["notes"] = "Intervals/Tempo. Avoid back-to-back hard days."
                    used_quality = True
                else:
                    entry["type"] = "Easy"
                    entry["distance_miles"] = round(max(3.0,base_easy - 0.5), 1)

            rows.append(entry)

        #Strength on easy/off/recovery days, limited by your settings
        if strength_yes and strength_days_per_week > 0:
            week_rows_idx = [i for i, r in enumerate(rows) if week_start <= pd.to_datetime(r["Date"]) <= pd.to_datetime(week_start + timedelta(days = 6))]
            candidates = [i for i in week_rows_idx if rows[i]["type"] in ("Easy","Off","Recovery")]
            for idx in candidates[:strength_days_per_week]:
                rows[idx]["Notes"] = (rows[idx]["Notes"] + " ").strip() + "+ Strength"

        # Add race day entry at the end so it appears in the plan
    rows.append(
        {
        "date": race_day,
        "day": DAYS[pd.Timestamp(race_day).dayofweek],
        "type": "Race",
        "distance_miles": goal_distance_miles,
        "Notes" : "Good luck!"
    }
    )

    # Clean and sort the result
    plan = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Date"], keep ='last')
        .sort_values("Date")
    )
    return plan



    



    
