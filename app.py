#notes on imports:
# streamlit: makes the web UI
# pandas: data manipulation (Handles tables)
# numpy: numerical calculations (math functions)(we use np.nan for missing values)
# datetime: dates and simple date arithmetic
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime


# Streamlit UI
st.write(""" # My first app Hello World """)

df = pd.read_csv("data/running_data.csv")
st.line_chart(df)
