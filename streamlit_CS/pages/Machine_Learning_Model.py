import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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

# Extra guards: after dropna + filters we still need data and ≥ 2 classes
if df_ml.empty:
    st.warning("Not enough Pokémon with complete stat data after filtering to train a model.")
    st.stop()

if df_ml["primary_type"].nunique() < 2:
    st.warning("Not enough different primary types in the filtered data to train a classifier.")
    st.stop()

X = df_ml[STAT_COLS].values
y = df_ml["primary_type"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = list(label_encoder.classes_)

# For K-fold upper bound, we need class counts *from the filtered data*
class_counts = df_ml["primary_type"].value_counts()
min_class_count = int(class_counts.min())  # smallest class size

# Base K on the rarest class, capped at 10
max_k_allowed = min(10, min_class_count)

# Make sure K is at least 2 so StratifiedKFold works
if max_k_allowed < 2:
    max_k_allowed = 2

# ───────────────────────────
# ROW 1 – controls (col 1) + confusion matrix (col 2) + feature importances (col 3)
# ───────────────────────────
col1_r1, col2_r1, col3_r1 = st.columns([1, 2, 1])

with col1_r1:
    st.subheader("Random Forest Settings")

    if max_k_allowed == 2:
        k_folds = 2
        st.caption(
            "Using K = 2 (minimum allowed) because at least one primary type is very rare "
            "with the current filters."
        )
    else:
        k_folds = st.slider(
            "Number of Folds (K)",
            min_value=2,
            max_value=max_k_allowed,
            value=min(5, max_k_allowed),
            help=(
                "How many chunks (folds) the data is split into for cross-validation. "
                "Higher K means more training runs but a more stable estimate of accuracy."
            ),
        )

    n_estimators = st.slider(
        "Number of Trees (n_estimators)",
        50,
        500,
        200,
        step=10,
        help="How many decision trees are in the forest. More trees usually improve stability, "
             "but also take longer to train."
    )

    max_depth = st.slider(
        "Max Depth (None = unlimited)",
        1,
        50,
        15,
        help="How many splits each tree is allowed to make from top to bottom. "
             "Shallower trees are simpler and may generalize better."
    )

    use_max_depth_none = st.checkbox(
        "Disable max depth (use None)",
        value=False,
        help="If checked, trees can grow as deep as they want until other stopping rules are hit."
    )

    min_samples_split = st.slider(
        "Min Samples Split",
        2,
        20,
        2,
        help="The minimum number of data points required in a node before it can be split into children."
    )

    min_samples_leaf = st.slider(
        "Min Samples Leaf",
        1,
        10,
        1,
        help="The minimum number of data points that must end up in a leaf node "
             "(the final box at the bottom of a tree)."
    )

    criterion = st.selectbox(
        "Split Criterion",
        ["gini", "entropy"],
        index=0,
        help="The formula used to measure how 'mixed' a node is. "
             "Both try to create purer groups of types after each split."
    )

    max_features_choice = st.selectbox(
    "Max Features per Split",
    ["sqrt", "log2", "None"],
    index=0,
    help="How many stats are considered at each split. 'sqrt' and 'log2' introduce randomness. "
         "'None' means all features are always used."
    )

    if max_features_choice == "None":
        rf_max_features = None
    else:
        rf_max_features = max_features_choice

    bootstrap = st.checkbox(
        "Use Bootstrap Samples",
        value=True,
        help="If checked, each tree is trained on a random sample (with replacement) of the data. "
             "This is part of what makes a random forest work well."
    )

    if use_max_depth_none:
        rf_max_depth = None
    else:
        rf_max_depth = max_depth

# ───────────────────────────
# TRAIN MODEL WITH STRATIFIED K-FOLD
# ───────────────────────────
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
        max_features=rf_max_features,
        bootstrap=bootstrap,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    cm_total += cm

# Fit one model on the full data for feature importances
rf_full = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    max_features=rf_max_features,
    bootstrap=bootstrap,
    n_jobs=-1,
)
rf_full.fit(X, y_encoded)

importances = rf_full.feature_importances_
feat_imp = (
    pd.DataFrame({"feature": STAT_COLS, "importance": importances})
    .sort_values("importance", ascending=False)
)

mean_acc = float(np.mean(fold_accuracies))

with col2_r1:
    st.markdown(
        f"<h2 style='text-align:center; margin-top: 0.5rem;'>"
        f"* Model Accuracy: {mean_acc * 100:.2f}% *"
        f"</h2>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
    "<h3 style='text-align:center;'>Confusion Matrix</h3>",
    unsafe_allow_html=True
    )

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
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(cm_fig, use_container_width=True)

with col3_r1:
    st.subheader("Feature Importances")

    fig_imp = px.bar(
        feat_imp,
        x="importance",
        y="feature",
        orientation="h",
        title="Relative Importance of Stats",
        text_auto=".2f",
    )

    fig_imp.update_layout(
        xaxis_title="Importance",
        yaxis_title="Stat",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# ───────────────────────────
# ROW 2 – Decision Tree Visualization (simple filled tree at max depth=3)
# ───────────────────────────
from sklearn.tree import plot_tree

st.divider()
st.subheader("Example Decision Tree from the Random Forest (Max Depth = 3 for display)")

# Fit a Random Forest using current hyperparameters (unchanged)
rf_viz = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    max_features=rf_max_features,
    bootstrap=bootstrap,
    n_jobs=-1,
)

rf_viz.fit(X, y_encoded)

# Extract one tree
tree_clf = rf_viz.estimators_[0]

# Visualize only top part of the tree: max depth = 3
fig, ax = plt.subplots(figsize=(22, 12))

plot_tree(
    tree_clf,
    feature_names=STAT_COLS,
    class_names=class_names,
    filled=True,       # sklearn colors based on class majority
    rounded=True,
    impurity=True,
    fontsize=10,
    max_depth=3,       # << limit display depth to 3
    ax=ax,
)

plt.tight_layout()
st.pyplot(fig)

# ───────────────────────────
# FOOTER
# ───────────────────────────
st.divider()
st.caption("**Data source:** https://pokeapi.co")
