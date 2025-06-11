import streamlit as st
import pandas as pd
from functions import *
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load and display logo in the top-left
from PIL import Image
logo = Image.open("logo.png")

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)  # Adjust width as needed
st.markdown("<small>Created by Ibrahim Oksuzoglu</small>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    shots_normalised_df = pd.read_pickle('shots_normalised_df.pkl')
    match_statistics = pd.read_pickle('match_statistics.pkl')
    return shots_normalised_df, match_statistics

shots_normalised_df, match_statistics = load_data()

teams = sorted(shots_normalised_df['h_team'].dropna().unique())

st.markdown("<h1 style='text-align: center;'>⚽ Match Shot Analysis Dashboard</h1>", unsafe_allow_html=True)

# Team selectors on top
col1_select, col2_select = st.columns(2)

with col1_select:
    home_team = st.selectbox(
        "🔴 Select Home Team",
        options=teams,
        index=teams.index("Arsenal") if "Arsenal" in teams else 0
    )

with col2_select:
    away_team = st.selectbox(
        "🔵 Select Away Team",
        options=teams,
        index=teams.index("Manchester City") if "Manchester City" in teams else 1 if len(teams) > 1 else 0
    )


if home_team == away_team:
    st.info("❗ Please select different teams for Home and Away.")

st.markdown("""
            <div style='display: flex; align-items: center;'>
            <hr style='flex: 1; border: none; border-top: 1px solid #ccc; margin-right: 10px;' />
            <h3 style='margin: 0; white-space: nowrap;'>🎯Shot map</h3>
            <hr style='flex: 1; border: none; border-top: 1px solid #ccc; margin-left: 10px;' />
            </div>
            """, unsafe_allow_html=True)

# Top row: 3 columns with shots plot in the center
col1, col2, col3 = st.columns([3, 6, 3])
with col1:
    tabs = st.tabs(["🥅 Goals", "📈 xG", "🎯 Shots"])
    
    with tabs[0]:  # Goals tab
        if home_team and away_team and home_team != away_team:
            st.write(f"🏠 **{home_team} Home Goal Distribution**")
            plot_home_goal_boxplot(match_statistics, home_team, away_team)
            st.pyplot(plt.gcf())

    with tabs[1]:  # xG tab
        if home_team and away_team and home_team != away_team:
            st.write(f"📊 **{home_team} Home xG Distribution**")
            plot_home_xG_boxplot(shots_normalised_df, match_statistics, home_team, away_team)
            st.pyplot(plt.gcf())

    with tabs[2]:  # Shots tab
        if home_team and away_team and home_team != away_team:
            st.write(f"🎯 **{home_team} Home Shots Distribution**")
            plot_home_shots_boxplot(shots_normalised_df, home_team, away_team)
            st.pyplot(plt.gcf())
with col2:
    if home_team and away_team:
        if home_team != away_team:
            plot_match_shots(shots_normalised_df, match_statistics, home_team, away_team)
            st.pyplot(plt.gcf())
with col3:
        tabs = st.tabs(["🥅 Goals", "📈 xG", "🎯 Shots"])

        with tabs[0]:  # Goals tab
            if home_team and away_team and home_team != away_team:
                st.write(f"🛫 **{away_team} Away Goal Distribution**")
                plot_away_goal_boxplot(match_statistics, home_team, away_team)
                st.pyplot(plt.gcf())

        with tabs[1]:  # xG tab
            if home_team and away_team and home_team != away_team:
                st.write(f"📊 **{away_team} Away xG Distribution**")
                plot_away_xG_boxplot(shots_normalised_df, match_statistics, home_team, away_team)
                st.pyplot(plt.gcf())

        with tabs[2]:  # Shots tab
            if home_team and away_team and home_team != away_team:
                st.write(f"🎯 **{away_team} Away Shots Distribution**")
                plot_away_shots_boxplot(shots_normalised_df, home_team, away_team)
                st.pyplot(plt.gcf())
        
if home_team and away_team:
    if home_team != away_team:
        st.markdown("""
                    <div style='display: flex; align-items: center;'>
                    <hr style='flex: 1; border: none; border-top: 1px solid #ccc; margin-right: 10px;' />
                    <h3 style='margin: 0; white-space: nowrap;'>🔍 Raw Shots Data</h3>
                    <hr style='flex: 1; border: none; border-top: 1px solid #ccc; margin-left: 10px;' />
                    </div>
                    """, unsafe_allow_html=True)

        if home_team != away_team:
            filtered_df = filtered_shot_dataframe(shots_normalised_df, home_team, away_team)
            st.dataframe(filtered_df, height=550)