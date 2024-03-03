# Specify the bowler
import pandas as pd
import numpy as np



def simulate_outcome(specific_striker, specific_bowler):
    df = pd.read_csv("odi_match_data2.csv")
    # Filter for the specific striker
    df_striker = df[df['striker'] == specific_striker]

    # Calculate runs scored distribution
    runs_distribution = df_striker['runs_off_bat'].value_counts(normalize=True).sort_index()
    probabilities = runs_distribution.values

    # Possible run outcomes
    run_values = runs_distribution.index

    # Filter for the specific bowler
    df_bowler = df[df['bowler'] == specific_bowler]

    # Calculate runs conceded distribution
    runs_conceded_distribution = df_bowler['runs_off_bat'].value_counts(normalize=True).sort_index()

    run_values_bowler = runs_conceded_distribution.index

    # Ensure both batsman and bowler have probabilities for the same run values
    # Extended one or both arrays to include missing run values with a 0 probability
    all_run_values = sorted(set(run_values) | set(run_values_bowler))
    all_probabilities_batsman = np.array([runs_distribution.get(run, 0) for run in all_run_values])
    all_probabilities_bowler = np.array([runs_conceded_distribution.get(run, 0) for run in all_run_values])

    # Average the probabilities
    average_probabilities = (all_probabilities_batsman + all_probabilities_bowler) / 2

    # Normalize the averaged probabilities
    normalized_probabilities = average_probabilities / average_probabilities.sum()

    # Simulate one outcome using the averaged and normalized probabilities
    simulated_run_averaged = np.random.choice(all_run_values, p=normalized_probabilities)

    return simulated_run_averaged


