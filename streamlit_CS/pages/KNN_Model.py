import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

from filters import apply_pokemon_filters, TYPE_COLORS  # TYPE_COLORS not used yet, but available

# ───────────────────────────
# PAGE CONFIG
# ───────────────────────────
st.set_page_config(
    page_title="KNN Model",
    page_icon="📍",
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
st.title("K-Nearest Neighbors Model")
st.markdown(
    "This page trains a **K-Nearest Neighbors (KNN)** classifier to predict a Pokémon’s "
    "primary type from its six base stats. We use stratified K-fold cross-validation "
    "to estimate how well the model generalizes."
)
st.caption(f"Current filters: {len(df_filtered)} Pokémon selected.")
st.divider()

# Guard: need data and at least 2 classes
if df_filtered.empty:
    st.warning("No Pokémon available for the selected filters. Adjust filters in the sidebar.")
    st.stop()

if df_filtered["primary_type"].nunique() < 2:
    st.warning("Not enough primary type classes in the filtered data to train a classifier.")
    st.stop()

# ───────────────────────────
# FEATURES AND TARGET
# ───────────────────────────
STAT_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

df_ml = df_filtered.dropna(subset=STAT_COLS + ["primary_type"]).copy()
X = df_ml[STAT_COLS].values
y = df_ml["primary_type"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = list(label_encoder.classes_)

# For K-fold upper bound, we need class counts
class_counts = pd.Series(y).value_counts()
min_class_count = int(class_counts.min())
max_k_allowed = max(2, min(10, min_class_count))

# ───────────────────────────
# ROW 1 – controls (col 1) + confusion matrix (col 2)
# ───────────────────────────
col1_r1, col2_r1 = st.columns([1, 3])

with col1_r1:
    st.subheader("KNN Settings")

    k_folds = st.slider(
        "Number of Folds (K)",
        min_value=2,
        max_value=max_k_allowed,
        value=min(5, max_k_allowed),
        help="How many chunks (folds) the data is split into for cross-validation. "
             "Higher K means more training runs but a more stable estimate of accuracy.",
    )

    n_neighbors = st.slider(
        "Number of Neighbors (k)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="How many nearby Pokémon (neighbors) are used to decide the predicted type."
    )

    weights = st.selectbox(
        "Neighbor Weights",
        ["uniform", "distance"],
        index=0,
        help="`uniform` treats all neighbors equally. `distance` gives closer neighbors more influence."
    )

    metric = st.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan"],
        index=0,
        help="How distance between Pokémon is measured in stat space."
    )

# ───────────────────────────
# TRAIN MODEL WITH STRATIFIED K-FOLD
# ───────────────────────────
kf = StratifiedKFold(n_splits=k_folds, shuffle=True)

fold_accuracies = []
cm_total = np.zeros((len(class_names), len(class_names)), dtype=int)

for train_idx, test_idx in kf.split(X, y_encoded):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Standardize stats + KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

    model = make_pipeline(StandardScaler(), knn)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    cm_total += cm

mean_acc = float(np.mean(fold_accuracies))

with col2_r1:
    st.markdown(
        f"<h3 style='text-align:center; margin-top: 0.5rem;'>"
        f"Mean Cross-Validated Accuracy: {mean_acc * 100:.2f}%"
        f"</h3>",
        unsafe_allow_html=True,
    )

    st.subheader("Confusion Matrix")

    cm_fig = px.imshow(
        cm_total,
        x=class_names,
        y=class_names,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(color="Count", x="Predicted Primary Type", y="True Primary Type"),
    )

    cm_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(cm_fig, use_container_width=True)

# ───────────────────────────
# ROW 2 – simple accuracy summary
# ───────────────────────────
st.divider()
st.subheader("Model Accuracy Summary")
st.write(f"Mean cross-validated accuracy over {k_folds} folds: **{mean_acc * 100:.2f}%**")

# ───────────────────────────
# FOOTER
# ───────────────────────────
st.divider()
st.caption("**Data source:** https://pokeapi.co")
