import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ───────────────────────────
# PAGE CONFIG
# ───────────────────────────
st.set_page_config(
    page_title="Introduction to Pokémon Types & Stats",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────
# LOAD DATA
# ───────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "pokemon_dataset.csv"
df = pd.read_csv(DATA_PATH)

TYPE_ICON_DIR = Path(__file__).parent.parent / "images" / "types"
POKEMON_IMG_DIR = Path(__file__).parent.parent / "images" / "pokemon"

TYPE_ORDER = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

STAT_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

STAT_COLORS = {
    "HP": "lime",
    "Attack": "yellow",
    "Defense": "orange",
    "Special Attack": "cyan",
    "Special Defense": "blue",
    "Speed": "magenta",
}

# ───────────────────────────
# HELPER: GET STATS FOR A SPECIES
# ───────────────────────────
def get_pokemon_stats(df: pd.DataFrame, pokemon_id: int) -> pd.DataFrame:
    row = df.loc[df["pokemon_id"] == pokemon_id].iloc[0]

    stats = {
        "HP": row["hp"],
        "Attack": row["attack"],
        "Defense": row["defense"],
        "Special Attack": row["special-attack"],
        "Special Defense": row["special-defense"],
        "Speed": row["speed"],
    }

    return pd.DataFrame({"Stat": list(stats.keys()), "Value": list(stats.values())})


def stat_bar_chart(stats_df: pd.DataFrame, title: str):
    fig = px.bar(
        stats_df,
        x="Value",
        y="Stat",
        orientation="h",
        color="Stat",
        color_discrete_map=STAT_COLORS,
        title=title,
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title="Base Stat",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        height=250,
    )
    return fig

# ───────────────────────────
# PAGE BODY
# ───────────────────────────
st.title("📘 Introduction to Pokémon Types and Stats")

st.markdown("""
Pokémon battles revolve around **types** and **stats**.

Every Pokémon has a primary type and may also have a secondary type (these Pokémon are often referred to has dual-typed).  
Types interact through strengths and weaknesses (for example, Water is strong against Fire, but weak to Electric and Grass).  
Understanding how types combine and how stats differ across Pokémon is essential to both competitive play and model building.
""")

st.markdown("### Pokémon Types")

# 2×9 Grid of type icons
row1_types = TYPE_ORDER[:9]
row2_types = TYPE_ORDER[9:]

cols_row1 = st.columns(9)
for col, t in zip(cols_row1, row1_types):
    with col:
        icon_path = TYPE_ICON_DIR / f"{t}.png"
        st.image(icon_path, use_column_width=True)
        st.markdown(
            f"<div style='text-align:center;'>{t.capitalize()}</div>",
            unsafe_allow_html=True,
        )

cols_row2 = st.columns(9)
for col, t in zip(cols_row2, row2_types):
    with col:
        icon_path = TYPE_ICON_DIR / f"{t}.png"
        st.image(icon_path, use_column_width=True)
        st.markdown(
            f"<div style='text-align:center;'>{t.capitalize()}</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

st.markdown("### Pokémon Stats")

st.markdown("""
Every Pokémon has six base stats that describe its general strengths:

- **HP (Hit Points):** Determines how much damage a Pokémon can take before fainting.
- **Attack:** Informally known as Physical Attack. Determines how much damage a Pokémon deals when using a physical move.
- **Defense:** Informally known as Physical Defense. Determines how much damage a Pokémon receives when hit with a physical move.
- **Special Attack:** Determines how much damage a Pokémon deals when using a special move.
- **Special Defense:** Determines how much damage a Pokémon receives when hit with a special move.
- **Speed:** Determines move order in battle (higher Speed usually attacks first).

These stats are what our machine learning model uses to try to predict a Pokémon’s primary type.
""")

st.markdown("---")

# ───────────────────────────
# THREE POKÉMON EXAMPLES WITH IMAGES + STAT BARS
# ───────────────────────────
st.markdown("### Example Pokémon and Their Stats")

col_pika, col_weezing, col_char = st.columns(3)

# Pikachu
with col_pika:
    st.markdown(
        "<h3 style='text-align:center;'>Pikachu</h3>",
        unsafe_allow_html=True,
    )

    pika_img = POKEMON_IMG_DIR / "pikachu.png"
    img_left, img_center, img_right = st.columns([1, 2, 1])
    with img_center:
        if pika_img.exists():
            st.image(pika_img, width=160)
        else:
            st.caption("Add image at `images/pokemon/pikachu.png`")

    pika_stats = get_pokemon_stats(df, 25)  # Pikachu = ID 25
    if pika_stats is not None:
        fig_pika = stat_bar_chart(pika_stats, "Pikachu – Base Stats")
        st.plotly_chart(fig_pika, use_container_width=True)
    else:
        st.warning("Could not find Pikachu in the dataset.")

# Weezing
with col_weezing:
    st.markdown(
        "<h3 style='text-align:center;'>Weezing</h3>",
        unsafe_allow_html=True,
    )

    weezing_img = POKEMON_IMG_DIR / "weezing.png"
    img_left, img_center, img_right = st.columns([1, 2, 1])
    with img_center:
        if weezing_img.exists():
            st.image(weezing_img, width=160)
        else:
            st.caption("Add image at `images/pokemon/weezing.png`")

    weezing_stats = get_pokemon_stats(df, 110)  # Weezing = ID 110
    fig_weezing = stat_bar_chart(weezing_stats, "Weezing – Base Stats")
    st.plotly_chart(fig_weezing, use_container_width=True)

# Charizard
with col_char:
    st.markdown(
        "<h3 style='text-align:center;'>Charizard</h3>",
        unsafe_allow_html=True,
    )

    char_img = POKEMON_IMG_DIR / "charizard.png"
    img_left, img_center, img_right = st.columns([1, 2, 1])
    with img_center:
        if char_img.exists():
            st.image(char_img, width=160)
        else:
            st.caption("Add image at `images/pokemon/charizard.png`")

    char_stats = get_pokemon_stats(df, 6)  # Charizard = ID 6
    fig_char = stat_bar_chart(char_stats, "Charizard – Base Stats")
    st.plotly_chart(fig_char, use_container_width=True)
