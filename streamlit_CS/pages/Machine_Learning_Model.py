import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
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
col1_r1, col2_r1 = st.columns([1, 2])

with col1_r1:
    st.subheader("Random Forest Settings")

    k_folds = st.slider(
        "Number of Folds (K)",
        min_value=2,
        max_value=max_k_allowed,
        value=min(5, max_k_allowed),
        help="Number of folds for Stratified K-Fold cross-validation.",
    )

    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, step=10)
    max_depth = st.slider("Max Depth (None = unlimited)", 1, 50, 15)
    use_max_depth_none = st.checkbox("Disable max depth (use None)", value=False)

    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

    criterion = st.selectbox("Split Criterion", ["gini", "entropy"], index=0)
    max_features = st.selectbox("Max Features per Split", ["sqrt", "log2", "auto"], index=0)

    bootstrap = st.checkbox("Use Bootstrap Samples", value=True)

    if use_max_depth_none:
        rf_max_depth = None
    else:
        rf_max_depth = max_depth

# ───────────────────────────
# Train single RF for visualization
# ───────────────────────────
viz_rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    max_features=max_features if max_features != "auto" else "auto",
    bootstrap=bootstrap,
    n_jobs=-1,
)

viz_rf.fit(X, y_encoded)

# ───────────────────────────
# TRAIN MODEL WITH STRATIFIED K-FOLD
# ───────────────────────────
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=k_folds, shuffle=True)

fold_accuracies = []
cm_total = np.zeros((len(class_names), len(class_names)), dtype=int)

for train_idx, test_idx in kf.split(X, y_encoded):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features if max_features != "auto" else "auto",
        bootstrap=bootstrap,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    cm_total += cm

mean_acc = float(np.mean(fold_accuracies))

with col2_r1:
    st.subheader("Confusion Matrix (Aggregated Across Folds)")

    cm_fig = px.imshow(
        cm_total,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(color="Count", x="Predicted Type", y="True Type"),
    )

    cm_fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(cm_fig, use_container_width=True)

# ───────────────────────────
# ROW 2 – simple accuracy print
# ───────────────────────────
st.divider()
st.subheader("Model Accuracy Summary")
st.write(f"Mean cross-validated accuracy over {k_folds} folds: **{mean_acc * 100:.2f}%**")

# ───────────────────────────
# ROW 3 – Decision Tree Visualization
# ───────────────────────────
st.divider()
st.subheader("Example Decision Tree from the Random Forest")

# Fit a separate RF on all filtered data using the current hyperparameters
rf_viz = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    max_features=max_features if max_features != "auto" else "auto",
    bootstrap=bootstrap,
    n_jobs=-1,
)

rf_viz.fit(X, y_encoded)

# Take the first tree for visualization
tree_clf = rf_viz.estimators_[0]
tree_ = tree_clf.tree_

fig, ax = plt.subplots(figsize=(20, 10))

# Plot once, then recolor nodes
artists = plot_tree(
    tree_clf,
    feature_names=STAT_COLS,
    class_names=class_names,
    filled=True,        # we'll override the colors manually
    rounded=True,
    impurity=True,
    ax=ax,
)

# tree_.value has shape (n_nodes, 1, n_classes)
node_values = tree_.value

for node_index, artist in enumerate(artists):
    # Only recolor the node boxes (FancyBboxPatch)
    if not isinstance(artist, plt.matplotlib.patches.FancyBboxPatch):
        continue

    counts = node_values[node_index][0]  # class counts at this node
    total = counts.sum()
    if total == 0:
        continue

    purity = counts.max() / total  # 0–1, 1 = perfectly pure
    pred_class_idx = counts.argmax()
    pred_class_name = class_names[pred_class_idx]

    # Base color from the Pokémon type palette
    base_hex = TYPE_COLORS.get(pred_class_name, "#808080")
    base_rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])

    # Blend toward white based on (1 - purity)
    # purity = 1 → pure type color
    # purity = 0.5 → halfway to white
    blended_rgb = white * (1 - purity) + base_rgb * purity

    artist.set_facecolor(blended_rgb)
    artist.set_edgecolor(base_hex)
    artist.set_linewidth(2)

st.pyplot(fig)

# ───────────────────────────
# FOOTER
# ───────────────────────────
st.divider()
st.caption("**Data source:** https://pokeapi.co")
