#notes on imports:
# streamlit: makes the web UI
# pandas: data manipulation (Handles tables)
# numpy: numerical calculations (math functions)(we use np.nan for missing values)
# datetime: dates and simple date arithmetic
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from pathlib import Path


# Streamlit UI
st.write(""" # My first app Hello World """)

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "data" / "running_data.csv"   # -> <project>/data/running_data.csv
df = pd.read_csv(csv_path)
st.line_chart(df)
