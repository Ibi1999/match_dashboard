import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import seaborn as sns

def plot_match_shots(shots_normalised_df, match_statistics, h_team, a_team):
    """
    Plots both teams' shots on a full horizontal pitch:
    - Home team attacking left (left to right), away team attacking right (right to left) swapped
    - All shots black except goals: home goals red, away goals blue (swapped colors)
    - Facecolors have 50% opacity, edges fully opaque black
    - Displays key match statistics at center pitch with stat names in the middle,
      home stats on left in red, away stats on right in blue.
      Additionally, prints team names above home and away stats with white outline.
    - Adds a key at the bottom showing that xG means the size of the circle.
    """

    # Clean team names in shots df
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip()

    # Filter shots for the match
    match_shots = shots_normalised_df[
        (shots_normalised_df['h_team'] == h_team) & (shots_normalised_df['a_team'] == a_team)
    ].copy()

    if match_shots.empty:
        print(f"No shots found for match: {h_team} vs {a_team}")
        return

    # Convert to numeric & scale coordinates
    match_shots['X_final'] = pd.to_numeric(match_shots['X_final'], errors='coerce') * 120
    match_shots['Y_final'] = pd.to_numeric(match_shots['Y_final'], errors='coerce') * 80
    match_shots['xG'] = pd.to_numeric(match_shots['xG'], errors='coerce').fillna(0)
    match_shots.dropna(subset=['X_final', 'Y_final'], inplace=True)

    # Split into home and away shots
    home_shots = match_shots[match_shots['h_a'] == 'h'].copy()
    away_shots = match_shots[match_shots['h_a'] == 'a'].copy()

    # Sizes for shot markers
    min_size = 50
    home_sizes = np.clip(home_shots['xG'] * 1000, min_size, None)
    away_sizes = np.clip(away_shots['xG'] * 1000, min_size, None)

    # Define shot colors: home goals red, away goals blue; all else black
    def shot_color(row):
        if str(row['result']).lower() == 'goal':
            return 'red' if row['h_a'] == 'h' else 'blue'
        return 'black'

    home_colors = home_shots.apply(shot_color, axis=1)
    away_colors = away_shots.apply(shot_color, axis=1)

    # Function to add alpha to color
    def apply_alpha(color, alpha=0.3):
        rgba = list(mcolors.to_rgba(color))
        rgba[3] = alpha
        return tuple(rgba)

    home_colors_alpha = home_colors.apply(lambda c: apply_alpha(c, 0.5))
    away_colors_alpha = away_colors.apply(lambda c: apply_alpha(c, 0.5))

    # Flip X for correct direction
    home_shots['X_final'] = 120 - home_shots['X_final']
    away_shots['X_final'] = 120 - away_shots['X_final']

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#262730', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))

    # Plot home and away shots
    pitch.scatter(
        home_shots['X_final'], home_shots['Y_final'],
        s=home_sizes, ax=ax,
        facecolors=home_colors_alpha, edgecolors='black', linewidth=2,
        label=f"{h_team} shots"
    )
    pitch.scatter(
        away_shots['X_final'], away_shots['Y_final'],
        s=away_sizes, ax=ax,
        facecolors=away_colors_alpha, edgecolors='black', linewidth=2,
        label=f"{a_team} shots"
    )

    # Add team names at the top with white outline
    y_team = 10  # slightly above the pitch
    x_home_team = 120 * 0.20
    x_away_team = 120 * 0.80

    home_text = ax.text(
        x_home_team, y_team, h_team,
        ha='center', va='bottom', fontsize=26, color='red', fontweight='bold'
    )
    away_text = ax.text(
        x_away_team, y_team, a_team,
        ha='center', va='bottom', fontsize=26, color='blue', fontweight='bold'
    )

    for txt in [home_text, away_text]:
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])

    # Calculate xG sums from shots dataframe
    home_xg = home_shots['xG'].sum()
    away_xg = away_shots['xG'].sum()

    # Get goals from match_statistics
    match_stats_row = match_statistics[
        (match_statistics['h_team'] == h_team) & (match_statistics['a_team'] == a_team)
    ]

    if match_stats_row.empty:
        print(f"No match statistics found for {h_team} vs {a_team}")
        return

    home_goals = match_stats_row['Full Time Home Team Goals'].values[0]
    away_goals = match_stats_row['Full Time Away Team Goals'].values[0]

    # Define order and map for stats to display
    stats_order = [
        'Goals',
        'xG',
        'Shots',
        'Shots on Target',
        'Corners',
        'Fouls Committed',
        'Yellow Cards',
        'Red Cards'
    ]

    stat_map = {
        'Shots': 'Shots',
        'Shots on Target': 'Shots on Target',
        'Corners': 'Corners',
        'Fouls Committed': 'Fouls Committed',
        'Yellow Cards': 'Yellow Cards',
        'Red Cards': 'Red Cards'
    }

    # Y positions top to bottom
    y_positions = np.linspace(20, 60, len(stats_order))  # swapped 60,20 -> 20,60

    x_home, x_stat, x_away = 120 * 0.35, 60, 120 * 0.65

    for i, stat_name in enumerate(stats_order):
        if stat_name == 'Goals':
            home_val = home_goals
            away_val = away_goals
        elif stat_name == 'xG':
            home_val = round(home_xg, 2)
            away_val = round(away_xg, 2)
        else:
            home_col = 'Home Team ' + stat_map[stat_name]
            away_col = 'Away Team ' + stat_map[stat_name]
            home_val = match_stats_row[home_col].values[0]
            away_val = match_stats_row[away_col].values[0]

        ax.text(x_home, y_positions[i], str(home_val), ha='center', va='center',
                fontsize=20, color='red', fontweight='bold',
                path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        ax.text(x_away, y_positions[i], str(away_val), ha='center', va='center',
                fontsize=20, color='blue', fontweight='bold',
                path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        stat_text = ax.text(x_stat, y_positions[i], stat_name, ha='center', va='center',
                            fontsize=20, color='black', fontweight='bold',
                            path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
    
    legend_y = 75  # vertical position on pitch
    start_x = 40   # start x coordinate for the legend group

    # Plot black circle for shot
    ax.scatter(start_x, legend_y, s=100, c='black', edgecolors='black', linewidth=1)

    # Text next to black circle
    ax.text(start_x + 3, legend_y, '= Shot', ha='left', va='center',
            fontsize=14, color='white', fontweight='bold')

    # Plot red circle for goal
    ax.scatter(start_x + 25, legend_y, s=100, c='red', edgecolors='black', linewidth=1)

    # Text next to red circle
    ax.text(start_x + 28, legend_y, '= Goal', ha='left', va='center',
            fontsize=14, color='white', fontweight='bold')


    plt.show()

def plot_home_goal_density(match_statistics, h_team, a_team):
    # Filter all matches where the selected team played at home
    team_home_matches = match_statistics[match_statistics['h_team'] == h_team]

    # Get the number of goals they scored in this specific match
    match_row = match_statistics[
        (match_statistics['h_team'] == h_team) &
        (match_statistics['a_team'] == a_team)
    ]

    if match_row.empty:
        print(f"No match found for {h_team} vs {a_team}")
        return

    goals_this_match = match_row['Full Time Home Team Goals'].values[0]
    all_goals = team_home_matches['Full Time Home Team Goals']

    # Create the styled density plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Background colors
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')

    # Plot KDE (clipped to avoid negative x-axis)
    sns.kdeplot(
        all_goals, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_goals.max() + 1)  # avoid negatives
    )

    # Average line
    avg_goals = all_goals.mean()
    avg_line = ax.axvline(avg_goals, color='gray', linestyle='--', linewidth=1.5, label='Average')

    # Match-specific line
    match_line = ax.axvline(goals_this_match, color='red', linestyle='-', linewidth=2,
                            label=f'{h_team} vs {a_team}: {goals_this_match} goals')

    # Labels and title
    ax.set_title(f"{h_team}'s Home Goal Distribution", fontsize=14, color='white')
    ax.set_xlabel("Goals Scored", color='white')
    ax.set_ylabel("Density", color='white')

    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')

    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_away_goal_density(match_statistics, h_team, a_team):
    # Filter all matches where the selected team played away
    team_away_matches = match_statistics[match_statistics['a_team'] == a_team]

    # Get the number of goals they scored in this specific match
    match_row = match_statistics[
        (match_statistics['h_team'] == h_team) &
        (match_statistics['a_team'] == a_team)
    ]

    if match_row.empty:
        print(f"No match found for {h_team} vs {a_team}")
        return

    goals_this_match = match_row['Full Time Away Team Goals'].values[0]
    all_goals = team_away_matches['Full Time Away Team Goals']

    # Create the styled density plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Background colors
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')

    # Plot KDE (clipped to avoid negative x-axis)
    sns.kdeplot(
        all_goals, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_goals.max() + 1)  # avoid negatives
    )

    # Average line
    avg_goals = all_goals.mean()
    avg_line = ax.axvline(avg_goals, color='gray', linestyle='--', linewidth=1.5, label='Average')

    # Match-specific line
    match_line = ax.axvline(goals_this_match, color='blue', linestyle='-', linewidth=2,
                            label=f'{a_team} @ {h_team}: {goals_this_match} goals')

    # Labels and title
    ax.set_title(f"{a_team}'s Away Goal Distribution", fontsize=14, color='white')
    ax.set_xlabel("Goals Scored", color='white')
    ax.set_ylabel("Density", color='white')

    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')

    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_away_xG_density(shots_normalised_df, match_statistics, h_team, a_team):
    """
    Plots the KDE density of xG for the away team playing away,
    plus a vertical line for the specific match's xG value.
    """
    # Clean team names to be safe
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played away
    team_away_shots = shots_normalised_df[
        (shots_normalised_df['a_team'] == a_team) & (shots_normalised_df['h_a'] == 'a')
    ].copy()
    
    # Ensure xG is numeric
    team_away_shots['xG'] = pd.to_numeric(team_away_shots['xG'], errors='coerce')
    team_away_shots = team_away_shots.dropna(subset=['xG'])
    
    # Aggregate total xG per away match (group by h_team and a_team)
    agg_xg = team_away_shots.groupby(['h_team', 'a_team'])['xG'].sum().reset_index()
    
    # Get all aggregated xG values for this away team
    all_xg = agg_xg[agg_xg['a_team'] == a_team]['xG']
    
    if all_xg.empty:
        print(f"No xG data available for away team {a_team}")
        return
    
    # Find xG for the specific match h_team vs a_team
    match_xg_row = agg_xg[(agg_xg['h_team'] == h_team) & (agg_xg['a_team'] == a_team)]
    if match_xg_row.empty:
        print(f"No xG found for match {h_team} vs {a_team}")
        return
    xg_this_match = match_xg_row['xG'].values[0]
    
    # Plot KDE density
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.kdeplot(
        all_xg, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_xg.max() + 1)
    )
    
    # Average line
    avg_xg = all_xg.mean()
    ax.axvline(avg_xg, color='gray', linestyle='--', linewidth=1.5, label='Average xG')
    
    # Match-specific line
    ax.axvline(xg_this_match, color='blue', linestyle='-', linewidth=2,
               label=f'{a_team} @ {h_team}: {xg_this_match:.2f} xG')
    
    # Labels and title
    ax.set_title(f"{a_team}'s Away xG Distribution", fontsize=14, color='white')
    ax.set_xlabel("Expected Goals (xG)", color='white')
    ax.set_ylabel("Density", color='white')
    
    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_home_xG_density(shots_normalised_df, match_statistics, h_team, a_team):
    """
    Plots the KDE density of xG for the home team playing at home,
    plus a vertical line for the specific match's xG value.
    """
    # Clean team names for consistency
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played at home
    team_home_shots = shots_normalised_df[
        (shots_normalised_df['h_team'] == h_team) & (shots_normalised_df['h_a'] == 'h')
    ].copy()
    
    # Ensure xG is numeric and drop missing
    team_home_shots['xG'] = pd.to_numeric(team_home_shots['xG'], errors='coerce')
    team_home_shots = team_home_shots.dropna(subset=['xG'])
    
    # Aggregate total xG per home match (group by h_team and a_team)
    agg_xg = team_home_shots.groupby(['h_team', 'a_team'])['xG'].sum().reset_index()
    
    # Get all aggregated xG values for this home team
    all_xg = agg_xg[agg_xg['h_team'] == h_team]['xG']
    
    if all_xg.empty:
        print(f"No xG data available for home team {h_team}")
        return
    
    # Find xG for the specific match h_team vs a_team
    match_xg_row = agg_xg[(agg_xg['h_team'] == h_team) & (agg_xg['a_team'] == a_team)]
    if match_xg_row.empty:
        print(f"No xG found for match {h_team} vs {a_team}")
        return
    xg_this_match = match_xg_row['xG'].values[0]
    
    # Plot KDE density
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.kdeplot(
        all_xg, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_xg.max() + 1)
    )
    
    # Average line
    avg_xg = all_xg.mean()
    ax.axvline(avg_xg, color='gray', linestyle='--', linewidth=1.5, label='Average xG')
    
    # Match-specific line
    ax.axvline(xg_this_match, color='red', linestyle='-', linewidth=2,
               label=f'{h_team} vs {a_team}: {xg_this_match:.2f} xG')
    
    # Labels and title
    ax.set_title(f"{h_team}'s Home xG Distribution", fontsize=14, color='white')
    ax.set_xlabel("Expected Goals (xG)", color='white')
    ax.set_ylabel("Density", color='white')
    
    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_home_shots_density(shots_normalised_df, h_team, a_team):
    """
    Plots the KDE density of the total shots taken by the home team in all their home matches,
    plus a vertical line for the shots taken in the specific match.
    """
    # Clean team names for consistency
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played at home
    team_home_shots = shots_normalised_df[
        (shots_normalised_df['h_team'] == h_team) & (shots_normalised_df['h_a'] == 'h')
    ].copy()
    
    # Aggregate total shots per match (group by h_team and a_team)
    agg_shots = team_home_shots.groupby(['h_team', 'a_team']).size().reset_index(name='shots')
    
    # Get all shots values for this home team
    all_shots = agg_shots[agg_shots['h_team'] == h_team]['shots']
    
    if all_shots.empty:
        print(f"No shot data available for home team {h_team}")
        return
    
    # Find shots count for the specific match h_team vs a_team
    match_shots_row = agg_shots[(agg_shots['h_team'] == h_team) & (agg_shots['a_team'] == a_team)]
    if match_shots_row.empty:
        print(f"No shots found for match {h_team} vs {a_team}")
        return
    shots_this_match = match_shots_row['shots'].values[0]
    
    # Plot KDE density
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.kdeplot(
        all_shots, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_shots.max() + 1)
    )
    
    # Average line
    avg_shots = all_shots.mean()
    ax.axvline(avg_shots, color='gray', linestyle='--', linewidth=1.5, label='Average Shots')
    
    # Match-specific line
    ax.axvline(shots_this_match, color='red', linestyle='-', linewidth=2,
               label=f'{h_team} vs {a_team}: {shots_this_match} shots')
    
    # Labels and title
    ax.set_title(f"{h_team}'s Home Shots Distribution", fontsize=14, color='white')
    ax.set_xlabel("Number of Shots", color='white')
    ax.set_ylabel("Density", color='white')
    
    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_away_shots_density(shots_normalised_df, h_team, a_team):
    """
    Plots the KDE density of the total shots taken by the away team in all their away matches,
    plus a vertical line for the shots taken in the specific match.
    """
    # Clean team names for consistency
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played away
    team_away_shots = shots_normalised_df[
        (shots_normalised_df['a_team'] == a_team) & (shots_normalised_df['h_a'] == 'a')
    ].copy()
    
    # Aggregate total shots per match (group by h_team and a_team)
    agg_shots = team_away_shots.groupby(['h_team', 'a_team']).size().reset_index(name='shots')
    
    # Get all shots values for this away team
    all_shots = agg_shots[agg_shots['a_team'] == a_team]['shots']
    
    if all_shots.empty:
        print(f"No shot data available for away team {a_team}")
        return
    
    # Find shots count for the specific match h_team vs a_team
    match_shots_row = agg_shots[(agg_shots['h_team'] == h_team) & (agg_shots['a_team'] == a_team)]
    if match_shots_row.empty:
        print(f"No shots found for match {h_team} vs {a_team}")
        return
    shots_this_match = match_shots_row['shots'].values[0]
    
    # Plot KDE density
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.kdeplot(
        all_shots, fill=True, color='#b0b0b0', alpha=0.5, linewidth=2, ax=ax,
        clip=(0, all_shots.max() + 1)
    )
    
    # Average line
    avg_shots = all_shots.mean()
    ax.axvline(avg_shots, color='gray', linestyle='--', linewidth=1.5, label='Average Shots')
    
    # Match-specific line
    ax.axvline(shots_this_match, color='blue', linestyle='-', linewidth=2,
               label=f'{a_team} @ {h_team}: {shots_this_match} shots')
    
    # Labels and title
    ax.set_title(f"{a_team}'s Away Shots Distribution", fontsize=14, color='white')
    ax.set_xlabel("Number of Shots", color='white')
    ax.set_ylabel("Density", color='white')
    
    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_home_shots_boxplot(shots_normalised_df, h_team, a_team):
    """
    Plots a vertical boxplot of the total shots taken by the home team in all their home matches,
    and overlays a red dot for the shots taken in the specific match.
    """
    # Clean team names for consistency
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played at home
    team_home_shots = shots_normalised_df[
        (shots_normalised_df['h_team'] == h_team) & (shots_normalised_df['h_a'] == 'h')
    ].copy()
    
    # Aggregate total shots per match (group by h_team and a_team)
    agg_shots = team_home_shots.groupby(['h_team', 'a_team']).size().reset_index(name='shots')
    
    # Get all shots values for this home team
    all_shots = agg_shots[agg_shots['h_team'] == h_team]['shots']
    
    if all_shots.empty:
        print(f"No shot data available for home team {h_team}")
        return
    
    # Find shots count for the specific match h_team vs a_team
    match_shots_row = agg_shots[(agg_shots['h_team'] == h_team) & (agg_shots['a_team'] == a_team)]
    if match_shots_row.empty:
        print(f"No shots found for match {h_team} vs {a_team}")
        return
    shots_this_match = match_shots_row['shots'].values[0]
    
    # Plot vertical Boxplot
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.boxplot(y=all_shots, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)
    
    # Overlay match-specific point
    ax.scatter(0, shots_this_match, color='red', s=100, zorder=5,
               label=f'{h_team} vs {a_team}: {shots_this_match} shots')
    
    # Title and labels
    #ax.set_title(f"{h_team}'s Home Shots Distribution (Boxplot)", fontsize=14, color='white')
    ax.set_ylabel("Number of Shots", color='white')
    ax.set_xticks([])  # Hide x-axis since it's not meaningful here
    
    # # Legend
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def plot_away_shots_boxplot(shots_normalised_df, h_team, a_team):
    """
    Plots a vertical boxplot of the total shots taken by the away team in all their away matches,
    and overlays a blue dot for the shots taken in the specific match.
    """
    # Clean team names for consistency
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played away
    team_away_shots = shots_normalised_df[
        (shots_normalised_df['a_team'] == a_team) & (shots_normalised_df['h_a'] == 'a')
    ].copy()
    
    # Aggregate total shots per match (group by h_team and a_team)
    agg_shots = team_away_shots.groupby(['h_team', 'a_team']).size().reset_index(name='shots')
    
    # Get all shots values for this away team
    all_shots = agg_shots[agg_shots['a_team'] == a_team]['shots']
    
    if all_shots.empty:
        print(f"No shot data available for away team {a_team}")
        return
    
    # Find shots count for the specific match h_team vs a_team
    match_shots_row = agg_shots[(agg_shots['h_team'] == h_team) & (agg_shots['a_team'] == a_team)]
    if match_shots_row.empty:
        print(f"No shots found for match {h_team} vs {a_team}")
        return
    shots_this_match = match_shots_row['shots'].values[0]
    
    # Plot vertical Boxplot with updated size
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.boxplot(y=all_shots, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)
    
    # Overlay match-specific point
    ax.scatter(0, shots_this_match, color='blue', s=100, zorder=5,
               label=f'{a_team} @ {h_team}: {shots_this_match} shots')
    
    # Title and labels
    #ax.set_title(f"{a_team}'s Away Shots Distribution (Boxplot)", fontsize=13, color='white')
    ax.set_ylabel("Number of Shots", color='white')
    ax.set_xticks([])  # Hide x-axis since it's not meaningful here
    
    # # Legend
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def plot_home_xG_boxplot(shots_normalised_df, match_statistics, h_team, a_team):
    """
    Plots a vertical boxplot of total xG for the home team in all their home matches,
    and overlays a red dot for the specific match's xG value.
    """
    # Clean team names
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played at home
    team_home_shots = shots_normalised_df[
        (shots_normalised_df['h_team'] == h_team) & (shots_normalised_df['h_a'] == 'h')
    ].copy()
    
    # Ensure xG is numeric
    team_home_shots['xG'] = pd.to_numeric(team_home_shots['xG'], errors='coerce')
    team_home_shots = team_home_shots.dropna(subset=['xG'])
    
    # Aggregate total xG per home match
    agg_xg = team_home_shots.groupby(['h_team', 'a_team'])['xG'].sum().reset_index()
    
    all_xg = agg_xg[agg_xg['h_team'] == h_team]['xG']
    if all_xg.empty:
        print(f"No xG data available for home team {h_team}")
        return
    
    # Get match-specific xG
    match_xg_row = agg_xg[(agg_xg['h_team'] == h_team) & (agg_xg['a_team'] == a_team)]
    if match_xg_row.empty:
        print(f"No xG found for match {h_team} vs {a_team}")
        return
    xg_this_match = match_xg_row['xG'].values[0]
    
    # Plot vertical boxplot
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.boxplot(y=all_xg, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)
    
    # Overlay match-specific xG
    ax.scatter(0, xg_this_match, color='red', s=100, zorder=5,
               label=f'{h_team} vs {a_team}: {xg_this_match:.2f} xG')
    
    # Labels and title
    #ax.set_title(f"{h_team}'s Home xG Distribution (Boxplot)", fontsize=13, color='white')
    ax.set_ylabel("Expected Goals (xG)", color='white')
    ax.set_xticks([])
    
    # # Legend
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')
    
    # Styling
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def plot_away_xG_boxplot(shots_normalised_df, match_statistics, h_team, a_team):
    """
    Plots a vertical boxplot of total xG for the away team in all their away matches,
    and overlays a blue dot for the specific match's xG value.
    """
    # Clean team names
    shots_normalised_df['h_team'] = shots_normalised_df['h_team'].str.strip().str.title()
    shots_normalised_df['a_team'] = shots_normalised_df['a_team'].str.strip().str.title()
    
    # Filter shots where team played away
    team_away_shots = shots_normalised_df[
        (shots_normalised_df['a_team'] == a_team) & (shots_normalised_df['h_a'] == 'a')
    ].copy()
    
    # Ensure xG is numeric
    team_away_shots['xG'] = pd.to_numeric(team_away_shots['xG'], errors='coerce')
    team_away_shots = team_away_shots.dropna(subset=['xG'])
    
    # Aggregate total xG per away match
    agg_xg = team_away_shots.groupby(['h_team', 'a_team'])['xG'].sum().reset_index()
    
    all_xg = agg_xg[agg_xg['a_team'] == a_team]['xG']
    if all_xg.empty:
        print(f"No xG data available for away team {a_team}")
        return
    
    # Match-specific xG
    match_xg_row = agg_xg[(agg_xg['h_team'] == h_team) & (agg_xg['a_team'] == a_team)]
    if match_xg_row.empty:
        print(f"No xG found for match {h_team} vs {a_team}")
        return
    xg_this_match = match_xg_row['xG'].values[0]
    
    # Plot vertical boxplot
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')
    
    sns.boxplot(y=all_xg, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)
    
    # Overlay match-specific xG
    ax.scatter(0, xg_this_match, color='blue', s=100, zorder=5,
               label=f'{a_team} @ {h_team}: {xg_this_match:.2f} xG')
    
    # Labels and title
    #ax.set_title(f"{a_team}'s Away xG Distribution (Boxplot)", fontsize=13, color='white')
    ax.set_ylabel("Expected Goals (xG)", color='white')
    ax.set_xticks([])
    
    # # Legend styling
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')
    
    # Ticks and spines
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def plot_away_goal_boxplot(match_statistics, h_team, a_team):
    """
    Plots a vertical boxplot of full-time goals scored by the away team in all their away matches,
    and overlays a blue dot for the specific match's goal count.
    """
    # Filter all matches where the team played away
    team_away_matches = match_statistics[match_statistics['a_team'] == a_team]

    # Specific match goal count
    match_row = match_statistics[
        (match_statistics['h_team'] == h_team) &
        (match_statistics['a_team'] == a_team)
    ]

    if match_row.empty:
        print(f"No match found for {h_team} vs {a_team}")
        return

    goals_this_match = match_row['Full Time Away Team Goals'].values[0]
    all_goals = team_away_matches['Full Time Away Team Goals']

    # Plot vertical boxplot
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')

    sns.boxplot(y=all_goals, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)

    # Overlay dot for this match
    ax.scatter(0, goals_this_match, color='blue', s=100, zorder=5,
               label=f'{a_team} @ {h_team}: {goals_this_match} goals')

    # Labels and title
    #ax.set_title(f"{a_team}'s Away Goals (Boxplot)", fontsize=13, color='white')
    ax.set_ylabel("Goals Scored", color='white')
    ax.set_xticks([])

    # # Legend styling
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')

    # Ticks and spines
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def plot_home_goal_boxplot(match_statistics, h_team, a_team):
    """
    Plots a vertical boxplot of full-time goals scored by the home team in all their home matches,
    with a red dot indicating the number of goals in the specified match.
    """
    # Filter all matches where the selected team played at home
    team_home_matches = match_statistics[match_statistics['h_team'] == h_team]

    # Get the number of goals they scored in this specific match
    match_row = match_statistics[
        (match_statistics['h_team'] == h_team) &
        (match_statistics['a_team'] == a_team)
    ]

    if match_row.empty:
        print(f"No match found for {h_team} vs {a_team}")
        return

    goals_this_match = match_row['Full Time Home Team Goals'].values[0]
    all_goals = team_home_matches['Full Time Home Team Goals']

    # Create boxplot
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#262730')
    ax.set_facecolor('#262730')

    sns.boxplot(y=all_goals, color='#b0b0b0', width=0.4, ax=ax, fliersize=0)

    # Match-specific red dot
    ax.scatter(0, goals_this_match, color='red', s=100, zorder=5,
               label=f'{h_team} vs {a_team}: {goals_this_match} goals')

    # Title and labels
    # ax.set_title(f"{h_team}'s Home Goals (Boxplot)", fontsize=13, color='white')
    ax.set_ylabel("Goals Scored", color='white')
    ax.set_xticks([])

    # Legend styling
    # legend = ax.legend(loc='upper right', fontsize=5, facecolor='#262730', edgecolor='white', labelcolor='white')
    # for text in legend.get_texts():
    #     text.set_color('white')

    # Axis styling
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


def filtered_shot_dataframe(shots_normalised_df, h_team, a_team):
    """
    Filters the shots_normalised_df for rows where the home team and away team
    match the given inputs. Assumes team names may have inconsistent casing/whitespace.
    
    Parameters:
        shots_normalised_df (pd.DataFrame): The dataframe containing shot data.
        h_team (str): Home team name to filter by.
        a_team (str): Away team name to filter by.
    
    Returns:
        pd.DataFrame: Filtered dataframe with rows matching the teams.
    """
    # Clean team names in the dataframe for consistency
    df = shots_normalised_df.copy()
    df['h_team'] = df['h_team'].str.strip().str.title()
    df['a_team'] = df['a_team'].str.strip().str.title()

    # Clean input team names similarly
    h_team_clean = h_team.strip().title()
    a_team_clean = a_team.strip().title()

    # Filter rows based on teams
    filtered_df = df[(df['h_team'] == h_team_clean) & (df['a_team'] == a_team_clean)]

    return filtered_df
