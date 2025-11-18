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

def stat_boxplot(container, df_filtered, stat_col, stat_label):
    with container:
        st.subheader(f"{stat_label} Distribution by Primary Type")

        if df_filtered.empty:
            st.warning("No Pokémon available for the selected filters.")
        else:
            type_order = (
                df_filtered.groupby("primary_type")[stat_col]
                .mean()
                .reset_index()
                .sort_values(stat_col)["primary_type"]
                .tolist()
            )

            fig_box = px.box(
                df_filtered,
                x="primary_type",
                y=stat_col,
                category_orders={"primary_type": type_order},
                title=f"{stat_label} Stat Distribution by Primary Type",
                color="primary_type",
                color_discrete_map=TYPE_COLORS,
            )

            for trace in fig_box.data:
                t = trace.name
                c = TYPE_COLORS.get(t, "#808080")
                trace.update(
                    marker_color=c,
                    marker_line_color=c,
                    line_color=c,
                )

            fig_box.update_layout(
                xaxis_title="Primary Type",
                yaxis_title=stat_label,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )

            st.plotly_chart(fig_box, use_container_width=True)

# ───────────────────────────
# ROW 1
# ───────────────────────────
col1_r1, col2_r1 = st.columns(2)

with col1_r1:
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

with col2_r1:
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
# ROW 2 – HP, Attack
# ───────────────────────────
col1_r2, col2_r2 = st.columns(2)
stat_boxplot(col1_r2, df_filtered, "hp", "HP")
stat_boxplot(col2_r2, df_filtered, "attack", "Attack")

# ───────────────────────────
# ROW 3 – Defense, Special Attack
# ───────────────────────────
col1_r3, col2_r3 = st.columns(2)
stat_boxplot(col1_r3, df_filtered, "defense", "Defense")
stat_boxplot(col2_r3, df_filtered, "special-attack", "Special Attack")

# ───────────────────────────
# ROW 4 – Special Defense, Speed
# ───────────────────────────
col1_r4, col2_r4 = st.columns(2)
stat_boxplot(col1_r4, df_filtered, "special-defense", "Special Defense")
stat_boxplot(col2_r4, df_filtered, "speed", "Speed")

# ───────────────────────────
# ROW 5 – placeholders
# ───────────────────────────
col1_r5, col2_r5 = st.columns(2)

with col1_r5:
    st.subheader("Scatterplot Controls")

    stat_labels = {
        "HP": "hp",
        "Attack": "attack",
        "Defense": "defense",
        "Special Attack": "special-attack",
        "Special Defense": "special-defense",
        "Speed": "speed",
    }

    x_label = st.selectbox(
        "X-axis stat",
        list(stat_labels.keys()),
        index=0  # default HP
    )
    y_label = st.selectbox(
        "Y-axis stat",
        list(stat_labels.keys()),
        index=1  # default Attack
    )

    x_stat = stat_labels[x_label]
    y_stat = stat_labels[y_label]

    st.markdown("**Types to include**")

    selected_types = []
    for t in sorted(TYPE_COLORS.keys()):
        display_label = t.capitalize()
        default_checked = (t == "normal")
        if st.checkbox(display_label, value=default_checked, key=f"type_{t}"):
            selected_types.append(t)

with col2_r5:
    st.subheader("Stat vs Stat Scatterplot")

    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        df_scatter = df_filtered.copy()

        if selected_types:
            df_scatter = df_scatter[df_scatter["primary_type"].isin(selected_types)]
        else:
            df_scatter = df_scatter.iloc[0:0]

        if df_scatter.empty:
            st.warning("No Pokémon match the selected types.")
        else:
            fig_scatter = px.scatter(
                df_scatter,
                x=x_stat,
                y=y_stat,
                color="primary_type",
                color_discrete_map=TYPE_COLORS,
                title=f"{x_label} vs {y_label} by Primary Type",
            )

            fig_scatter.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                margin=dict(l=10, r=10, t=40, b=10),
                legend_title="Primary Type",
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
