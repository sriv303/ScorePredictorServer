import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from joblib import dump, load
# Load dataset
df = pd.read_csv("odi_match_data2.csv")

# Initialize a column for balls faced and determine if the player was dismissed
df['balls_faced'] = 1
df['is_out'] = np.where(df['striker'] == df['player_dismissed'], 1, 0)

# Calculate batsman statistics
batsman_grouped = df.groupby('striker').agg(total_balls=('balls_faced', 'sum'),
                                             dismissals=('is_out', 'sum'))
batsman_grouped['batsman_metric'] = batsman_grouped.apply(
    lambda x: x['total_balls'] / x['dismissals'] if x['dismissals'] > 0 else 0, axis=1)

# Handling bowler statistics with consideration for specific wickets
bowler_wickets_mask = df['wicket_type'].isin(['bowled', 'caught', 'lbw', 'hit wicket', 'caught and bowled', 'stumped'])
df['bowler_wickets'] = np.where(bowler_wickets_mask, 1, 0)

bowler_grouped = df.groupby('bowler').agg(
    total_runs_conceded=('runs_off_bat', 'sum'),
    balls_bowled=('balls_faced', 'sum'),
    wickets_taken=('bowler_wickets', 'sum')
)

# Calculating metrics with safe division
bowler_grouped['refined_bowler_strike_rate'] = bowler_grouped.apply(
    lambda x: x['balls_bowled'] / x['wickets_taken'] if x['wickets_taken'] > 0 else np.nan, axis=1)
bowler_grouped['bowler_econ'] = bowler_grouped['total_runs_conceded'] / bowler_grouped['balls_bowled'] * 6

# Handling potential infinite or NaN values after calculation
bowler_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
bowler_grouped.fillna({'refined_bowler_strike_rate': 0, 'bowler_econ': 0}, inplace=True)  # Assuming 0 for missing values

# Merging the calculated statistics back into the main dataframe
df = df.merge(bowler_grouped[['refined_bowler_strike_rate', 'bowler_econ']], how='left', left_on='bowler', right_index=True)
df = df.merge(batsman_grouped['batsman_metric'], how='left', left_on='striker', right_index=True)

# Preprocessing for modeling
features = ['venue', 'ball', 'striker', 'non_striker', 'bowler', 'batsman_metric', 'refined_bowler_strike_rate', 'bowler_econ']
X = df[features]
y_runs = df['runs_off_bat']
y_wicket = df['is_wicket'].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['venue', 'striker', 'non_striker', 'bowler']),
        ('num', StandardScaler(), ['ball', 'batsman_metric', 'refined_bowler_strike_rate', 'bowler_econ'])
    ])

# Split the data
X_train, X_test, y_train_runs, y_test_runs = train_test_split(X, y_runs, test_size=0.2, random_state=42)
X_train, X_test, y_train_wicket, y_test_wicket = train_test_split(X, y_wicket, test_size=0.2, random_state=42)

# Model pipelines
pipeline_runs = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

pipeline_wicket = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

now = datetime.now()

# Format the current time as a string (e.g., HH:MM:SS)
current_time = now.strftime("%H:%M:%S")

# Print the current time
print("Current time:", current_time)


# Fit the models
pipeline_runs.fit(X_train, y_train_runs)

now = datetime.now()
# Format the current time as a string (e.g., HH:MM:SS)
current_time = now.strftime("%H:%M:%S")
# Print the current time
print("Current time:", current_time)
dump(pipeline_runs, 'pipeline_runs.joblib')

pipeline_wicket.fit(X_train, y_train_wicket)

now = datetime.now()
# Format the current time as a string (e.g., HH:MM:SS)
current_time = now.strftime("%H:%M:%S")
# Print the current time
print("Current time:", current_time)
dump(pipeline_runs, 'pipeline_wicket.joblib')



# Simulate ball outcomes considering sparse data
def simulate_ball(striker, bowler, df):
    
    if is_data_sparse:
        batsman_metric = df[df['striker'] == striker]['batsman_metric'].mean()
        bowler_sr = df[df['bowler'] == bowler]['refined_bowler_strike_rate'].mean()
        bowler_econ = df[df['bowler'] == bowler]['bowler_econ'].mean() / 6  # Adjusting economy rate

        # Adjusting the lambda for poisson based on the new metrics
        predicted_runs = np.random.poisson(lam=(batsman_metric / 100) / bowler_econ)
        predicted_runs = min(predicted_runs, 6)  # Capping the runs at 6

        # Calculating wicket probability
        wicket_probability = 1 / ((1 / bowler_sr) + (1 / batsman_metric))
        predicted_wicket = np.random.rand() < wicket_probability
    else:
        input_data = df.loc[(df['striker'] == striker) & (df['bowler'] == bowler)].tail(1)
        predicted_runs = int(pipeline_runs.predict(input_data)[0])
        predicted_wicket = pipeline_wicket.predict(input_data)[0]

    return predicted_runs, int(predicted_wicket)
    

# Check for sparse data (function remains the same as provided earlier)
def is_data_sparse(striker, bowler, df):
    matches = df[(df['striker'] == striker) & (df['bowler'] == bowler)]
    return len(matches) < 10

# Simulate an over
def simulate_over(striker, non_striker, bowler, df):
    total_runs = 0
    wickets = 0

    for ball in range(1, 7):
        runs, wicket = simulate_ball(striker, bowler, df)
        print(f"Ball {ball}: Runs - {runs}, Wicket - {'Yes' if wicket else 'No'}")
        
        if wicket:
            wickets += 1
            total_runs += runs

            #break  # Stopping the simulation if a wicket is taken

        if runs % 2 != 0:  # Switching strikers on odd runs
            striker, non_striker = non_striker, striker

    return total_runs, wickets

# Example usage
striker = 'Shubman Gill'
non_striker = 'Virat Kohli'
bowler = 'JR Hazlewood'
simulate_over(striker, non_striker,  bowler, df)
