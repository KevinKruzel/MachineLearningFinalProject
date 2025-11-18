import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

TYPE_COLORS = {
    "normal":  "#A8A77A",
    "fire":    "#EE8130",
    "water":   "#6390F0",
    "electric":"#F7D02C",
    "grass":   "#7AC74C",
    "ice":     "#96D9D6",
    "fighting":"#C22E28",
    "poison":  "#A33EA1",
    "ground":  "#E2BF65",
    "flying":  "#A98FF3",
    "psychic": "#F95587",
    "bug":     "#A6B91A",
    "rock":    "#B6A136",
    "ghost":   "#735797",
    "dragon":  "#6F35FC",
    "dark":    "#705746",
    "steel":   "#B7B7CE",
    "fairy":   "#D685AD"
}

st.set_page_config(
    page_title="EDA Gallery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent.parent / "data" / "pokemon_dataset.csv"
df = pd.read_csv(DATA_PATH)

st.title("Exploratory Data Analysis Gallery")

st.sidebar.header("Filters")

include_legendary = st.sidebar.checkbox(
    "Include Legendary Pokémon",
    value=True
)

include_non_fully_evolved = st.sidebar.checkbox(
    "Include Non-Fully Evolved Pokémon",
    value=True
)

include_regional_variants = st.sidebar.checkbox(
    "Include Regional Variants",
    value=True
)

include_dual_typed = st.sidebar.checkbox(
    "Include Dual-Typed Pokémon",
    value=True
)

min_gen = int(df["generation"].min())
max_gen = int(df["generation"].max())

gen_min, gen_max = st.sidebar.slider(
    "Generation Range",
    min_value=min_gen,
    max_value=max_gen,
    value=(min_gen, max_gen),
    help="Filter Pokémon by generation number.",
)

df_filtered = df.copy()

df_filtered = df_filtered[
    (df_filtered["generation"] >= gen_min) &
    (df_filtered["generation"] <= gen_max)
]

if not include_legendary:
    df_filtered = df_filtered[df_filtered["is_legendary"] == False]

if not include_non_fully_evolved:
    df_filtered = df_filtered[df_filtered["is_fully_evolved"] == True]

if not include_regional_variants:
    df_filtered = df_filtered[df_filtered["is_regional_variant"] == False]

if not include_dual_typed:
    df_filtered = df_filtered[df_filtered["is_mono-type"] == True]

st.caption(f"Current filters: {len(df_filtered)} Pokémon selected.")

# ───────────────────────────
# ROW 1
# ───────────────────────────
big_col_r1_left, big_col_r1_right = st.columns([3, 2])

with big_col_r1_left:
    st.subheader("Type Combination Heatmap")

    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        temp = df_filtered.copy()

        temp["secondary_type_display"] = temp["secondary_type"]
        mono_mask = temp["secondary_type_display"].isna()
        temp.loc[mono_mask, "secondary_type_display"] = temp.loc[mono_mask, "primary_type"]

        type_counts = (
            temp.groupby(["primary_type", "secondary_type_display"])["pokemon_id"]
            .count()
            .reset_index(name="count")
        )

        pivot_table = type_counts.pivot_table(
            index="secondary_type_display",
            columns="primary_type",
            values="count",
            fill_value=0,
        )

        pivot_table = pivot_table.reindex(
            index=sorted(pivot_table.index),
            columns=sorted(pivot_table.columns),
        )

        fig = px.imshow(
            pivot_table,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(color="Number of Pokémon"),
        )
        
        fig.update_layout(
            xaxis_title="Primary Type",
            yaxis_title="Secondary Type",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

with big_col_r1_right:
    st.subheader("Number of Pokémon by Primary Type")

    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        type_counts_bar = (
            df_filtered.groupby("primary_type")["pokemon_id"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        fig_bar = px.bar(
            type_counts_bar,
            x="primary_type",
            y="count",
            title="Pokémon Count by Primary Type",
            color="primary_type",
            color_discrete_map=TYPE_COLORS,
            text_auto=True,
        )

        fig_bar.update_traces(textposition="outside")
        
        fig_bar.update_layout(
            xaxis_title="Primary Type",
            yaxis_title="Number of Pokémon",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig_bar, use_container_width=True)

# ───────────────────────────
# ROW 2
# ───────────────────────────
col1_r2, col2_r2, big_col_r2 = st.columns([1, 1, 3])

with col1_r2:
    st.subheader("Row 2 — Column 1")
    st.write("Placeholder text")

with col2_r2:
    st.subheader("Row 2 — Column 2")
    st.write("Placeholder text")

with big_col_r2:
    st.subheader("Attack Distribution by Primary Type")

    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        fig_box = px.box(
            df_filtered,
            x="primary_type",
            y="attack",
            color="primary_type",
            color_discrete_map=TYPE_COLORS,
            title="Attack Stat Distribution Grouped by Primary Type",
            points="outliers",
        )

        fig_box.update_layout(
            xaxis_title="Primary Type",
            yaxis_title="Attack",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig_box, use_container_width=True)

# ───────────────────────────
# ROW 3
# ───────────────────────────
col1_r3, col2_r3, big_col_r3 = st.columns([1, 1, 3])

with col1_r3:
    st.subheader("Row 3 — Column 1")
    st.write("Placeholder text.")

with col2_r3:
    st.subheader("Row 3 — Column 2")
    st.write("Placeholder text.")

with big_col_r3:
    st.subheader("Row 3 — Columns 3–5")
    st.write("Placeholder text.")

# ───────────────────────────
# ROW 4
# ───────────────────────────
col1_r4, col2_r4, big_col_r4 = st.columns([1, 1, 3])

with col1_r4:
    st.subheader("Row 4 — Column 1")
    st.write("Placeholder text.")

with col2_r4:
    st.subheader("Row 4 — Column 2")
    st.write("Placeholder text.")

with big_col_r4:
    st.subheader("Row 4 — Columns 3–5")
    st.write("Placeholder text.")

# ───────────────────────────
# ROW 5
# ───────────────────────────
col1_r5, col2_r5, big_col_r5 = st.columns([1, 1, 3])

with col1_r5:
    st.subheader("Row 5 — Column 1")
    st.write("Placeholder text.")

with col2_r5:
    st.subheader("Row 5 — Column 2")
    st.write("Placeholder text.")

with big_col_r5:
    st.subheader("Row 5 — Columns 3–5")
    st.write("Placeholder text.")

# ───────────────────────────
# ROW 6
# ───────────────────────────
col1_r6, col2_r6, big_col_r6 = st.columns([1, 1, 3])

with col1_r6:
    st.subheader("Row 6 — Column 1")
    st.write("Placeholder text.")

with col2_r6:
    st.subheader("Row 6 — Column 2")
    st.write("Placeholder text.")

with big_col_r6:
    st.subheader("Row 6 — Columns 3–5")
    st.write("Placeholder text.")

# ───────────────────────────
# ROW 7
# ───────────────────────────
col1_r7, col2_r7, big_col_r7 = st.columns([1, 1, 3])

with col1_r7:
    st.subheader("Row 7 — Column 1")
    st.write("Placeholder text.")

with col2_r7:
    st.subheader("Row 7 — Column 2")
    st.write("Placeholder text.")

with big_col_r7:
    st.subheader("Row 7 — Columns 3–5")
    st.write("Placeholder text.")

st.divider()

st.caption("**Data source:** https://pokeapi.co")

with st.expander("Data Preview"):
    st.dataframe(df)

csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Raw Data (CSV)",
    data=csv_data,
    file_name="pokemon_dataset.csv",
    mime="text/csv",
)
