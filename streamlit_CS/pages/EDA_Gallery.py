import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


st.set_page_config(
    page_title="EDA Gallery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent.parent / "data" / "pokemon_dataset.csv"
df = pd.read_csv(DATA_PATH)

st.title("Exploratory Data Analysis Gallery")

# ───────────────────────────
# SIDEBAR FILTERS
# ───────────────────────────
st.sidebar.header("Filters")

# ───────────────────────────
# ROW 1
# ───────────────────────────
big_col_r1, col3_r1 = st.columns([2, 1])

with big_col_r1:
    st.subheader("Row 1 — Column 1 and 2")
    st.write("Placeholder text")

with col3_r1:
    st.subheader("Row 1 — Column 3")
    st.write("Placeholder text")

# ───────────────────────────
# ROW 2
# ───────────────────────────
col1_r2, col2_r2, col3_r2 = st.columns(3)

with col1_r2:
    st.subheader("Row 2 — Column 1")
    st.write("Placeholder text")

with col2_r2:
    st.subheader("Row 2 — Column 2")
    st.write("Placeholder text")

with col3_r2:
    st.subheader("Row 2 — Column 3")
    st.write("Placeholder text")

# ───────────────────────────
# ROW 3
# ───────────────────────────
col1_r3, col2_r3, col3_r3 = st.columns(3)

with col1_r3:
    st.subheader("Row 3 — Column 1")
    st.write("Placeholder text.")

with col2_r3:
    st.subheader("Row 3 — Column 2")
    st.write("Placeholder text.")

with col3_r3:
    st.subheader("Row 3 — Column 3")
    st.write("Placeholder text.")

st.divider()

# Footer
st.caption("**Data source:** https://pokeapi.co")

with st.expander("Data Preview"):
    st.dataframe(df)

# Read the CSV file for download
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Raw Data (CSV)",
    data=csv_data,
    file_name="pokemon_dataset.csv",
    mime="text/csv",
)
