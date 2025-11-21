import streamlit as st

st.set_page_config(
    page_title="Machine Learning Final Project",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Machine Learning Final Project – Pokémon Type Prediction")
st.markdown("#### by Kevin Kruzel")

st.markdown(
    """
The goal of this project to see how well we can predict a Pokémon’s primary type using only its
six base stats (HP, Attack, Defense, Special Attack, Special Defense, and Speed) by using a machine learning model.

This app uses interactive tools to explore the dataset and customize the Random Forest model used to predict a Pokémon’s primary type.
Use the sidebar to filter the dataset to specific Pokémon to explore how these filters affect the distribution of the data and the accuracy of the model.
"""
)

st.divider()

st.markdown(
    """
Use the **sidebar** to switch between pages:

- 📊 **EDA Gallery**  - Explore the Pokémon dataset with interactive visualizations.  
  - View a type heatmap and bar chart to see the distribution of primary and secondary types of Pokemon
  - View a series of boxplots to visualize the differences in distribution of Pokémon stats grouped by type.
  - Create a fully customizable scatterplot to explore the distribution of two select stats for all Pokémon in select type groups.

<br>

- ⚙️ **Machine Learning Model**  - Build and evaluate a Random Forest model that predicts a Pokémon’s primary type.  
  - Adjust model hyperparameters (number of trees, depth, max features, etc.).
  - View a confusion matrix showing exactly what predictions the model is making.
  - See feature importance in a bar chart to understand which stats matter most.
  - Inspect an example decision tree from the forest to better understand the model.

<br>

Head over to the other pages in the sidebar to start exploring.
""",
    unsafe_allow_html=True,
)

st.divider()
st.caption("Built with Streamlit")
