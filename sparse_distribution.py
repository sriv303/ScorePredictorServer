# Specify the bowler
import pandas as pd
import numpy as np



def simulate_outcome(specific_striker, specific_bowler):
# Load the dataset
    df = pd.read_csv("odi_match_data2.csv")

    # Specify the striker
    #specific_striker = "RG Sharma"

    # Filter for the specific striker
    df_striker = df[df['striker'] == specific_striker]

    # Calculate runs scored distribution
    runs_distribution = df_striker['runs_off_bat'].value_counts(normalize=True).sort_index()

    # Assuming the distribution is complete, otherwise you need to ensure it sums to 1
    # Probabilities for scoring 0, 1, 2, ..., 6 runs
    probabilities = runs_distribution.values

    # Possible run outcomes
    run_values = runs_distribution.index

    # Simulate one outcome


    # Specify the bowler
    #specific_bowler = "JR Hazlewood"

    # Filter for the specific bowler
    df_bowler = df[df['bowler'] == specific_bowler]

    # Calculate runs conceded distribution
    runs_conceded_distribution = df_bowler['runs_off_bat'].value_counts(normalize=True).sort_index()

    # Assuming the distribution is complete, otherwise you need to ensure it sums to 1
    # Probabilities for conceding 0, 1, 2, ..., 6 runs
    #probabilities_bowler = runs_conceded_distribution.values

    # Possible run outcomes
    run_values_bowler = runs_conceded_distribution.index


    # Ensure both batsman and bowler have probabilities for the same run values
    # This might involve extending one or both arrays to include missing run values with a 0 probability
    all_run_values = sorted(set(run_values) | set(run_values_bowler))
    all_probabilities_batsman = np.array([runs_distribution.get(run, 0) for run in all_run_values])
    all_probabilities_bowler = np.array([runs_conceded_distribution.get(run, 0) for run in all_run_values])

    # Average the probabilities
    average_probabilities = (all_probabilities_batsman + all_probabilities_bowler) / 2

    # Normalize the averaged probabilities
    normalized_probabilities = average_probabilities / average_probabilities.sum()

    #print(f"Averaged and Normalized Probabilities: {normalized_probabilities}")


    # Simulate one outcome using the averaged and normalized probabilities
    simulated_run_averaged = np.random.choice(all_run_values, p=normalized_probabilities)
    #print(f"Simulated Run (Averaged): {simulated_run_averaged}")

    # Simulate 10 outcomes
    #simulated_runs_10_averaged = np.random.choice(all_run_values, size=10, p=normalized_probabilities)
    #print(f"Simulated Runs Over 10 Balls (Averaged): {simulated_runs_10_averaged}")
    
    return simulated_run_averaged


