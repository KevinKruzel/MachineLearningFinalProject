import streamlit as st
import pandas as pd

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

def apply_pokemon_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    include_legendary = st.sidebar.checkbox(
        "Include Legendary Pokémon",
        value=True,
        key="include_legendary"
    )

    include_non_fully_evolved = st.sidebar.checkbox(
        "Include Non-Fully Evolved Pokémon",
        value=True,
        key="include_non_fully_evolved"
    )

    include_regional_variants = st.sidebar.checkbox(
        "Include Regional Variants",
        value=True,
        key="include_regional_variants"
    )

    include_dual_typed = st.sidebar.checkbox(
        "Include Dual-Typed Pokémon",
        value=True,
        key="include_dual_typed"
    )

    min_gen = int(df["generation"].min())
    max_gen = int(df["generation"].max())

    gen_min, gen_max = st.sidebar.slider(
        "Generation Range",
        min_value=min_gen,
        max_value=max_gen,
        value=(min_gen, max_gen),
        help="Filter Pokémon by generation number.",
        key="generation_range",
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
    return df_filtered
