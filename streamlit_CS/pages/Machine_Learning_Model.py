import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from filters import apply_pokemon_filters
from filters import TYPE_COLORS

# ───────────────────────────
# PAGE CONFIG
# ───────────────────────────
st.set_page_config(
    page_title="Machine Learning Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────
# LOAD DATA
# ───────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "pokemon_dataset.csv"
df = pd.read_csv(DATA_PATH)

df_filtered = apply_pokemon_filters(df)

# ───────────────────────────
# PAGE HEADER
# ───────────────────────────
st.title("Machine Learning Model")

st.caption(f"Current filters: {len(df_filtered)} Pokémon selected.")

st.divider()

# ───────────────────────────
# ROW 1
# ───────────────────────────
col1_r1, col2_r1 = st.columns(2)

with col1_r1:
    st.subheader("Row 1 — Column 1")
    st.write("Placeholder text")

with col2_r1:
    st.subheader("Row 1 — Column 2")
    st.write("Placeholder text")

# ───────────────────────────
# FOOTER
# ───────────────────────────
st.divider()
st.caption("**Data source:** https://pokeapi.co")
