import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
df = pd.read_csv('cricket_data.csv')

# Calculate player statistics
df['balls'] = 1
batsman_runs = df.groupby('striker')['runs_off_bat'].sum()
balls_faced = df.groupby('striker')['balls'].sum()
batsman_avg = batsman_runs / balls_faced
batsman_sr = (batsman_runs / balls_faced) * 100

bowler_runs = df.groupby('bowler')['runs_off_bat'].sum()
balls_bowled = df.groupby('bowler')['balls'].sum()
wickets_taken = df.groupby('bowler')['is_wicket'].sum()
bowler_avg = bowler_runs / wickets_taken
bowler_sr = balls_bowled / wickets_taken
bowler_econ = (bowler_runs / balls_bowled) * 6

# Merge calculated statistics back into the dataframe
df = df.merge(batsman_avg.rename('batsman_avg'), how='left', left_on='striker', right_index=True)
df = df.merge(batsman_sr.rename('batsman_sr'), how='left', left_on='striker', right_index=True)
df = df.merge(bowler_avg.rename('bowler_avg'), how='left', left_on='bowler', right_index=True)
df = df.merge(bowler_sr.rename('bowler_sr'), how='left', left_on='bowler', right_index=True)
df = df.merge(bowler_econ.rename('bowler_econ'), how='left', left_on='bowler', right_index=True)

# Preprocess features and target for modeling
features = ['venue', 'ball', 'striker', 'non_striker', 'bowler', 'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_sr', 'bowler_econ']
X = df[features]
y_runs = df['runs_off_bat']
y_wicket = df['is_wicket'].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['venue', 'striker', 'non_striker', 'bowler']),
        ('num', StandardScaler(), ['ball', 'batsman_avg', 'batsman_sr', 'bowler_avg', 'bowler_sr', 'bowler_econ'])
    ])

# Train-test split
X_train, X_test, y_train_runs, y_test_runs = train_test_split(X, y_runs, test_size=0.2, random_state=42)
X_train, X_test, y_train_wicket, y_test_wicket = train_test_split(X, y_wicket, test_size=0.2, random_state=42)

# Model training
pipeline_runs = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
]).fit(X_train, y_train_runs)

pipeline_wicket = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
]).fit(X_train, y_train_wicket)

# Function to simulate a ball's outcome
def simulate_ball(striker, bowler, df):
    # Using averages as fallback if specific data is sparse
    if is_data_sparse(striker, bowler, df):
        batsman_avg = df[df['striker'] == striker]['batsman_avg'].mean()
        bowler_econ = df[df['bowler'] == bowler]['bowler_econ'].mean()
        predicted_runs = np.random.poisson(lam=(batsman_avg / bowler_econ))
        predicted_wicket = np.random.rand() < (1 / df[df['bowler'] == bowler]['bowler_sr'].mean())
    else:
        input_data = df.loc[(df['striker'] == striker) & (df['bowler'] == bowler)].tail(1)
        predicted_runs = int(pipeline_runs.predict(input_data)[0])
        predicted_wicket = pipeline_wicket.predict(input_data)[0]
    
    return predicted_runs, int(predicted_wicket)

# Function to check for sparse data
def is_data_sparse(striker, bowler, df):
    matches = df[(df['striker'] == striker) & (df['bowler'] == bowler)]
    return len(matches) < 5

# Function to simulate an over
def simulate_over(striker, non_striker, bowler, df):
    total_runs = 0
    wickets = 0

    for ball in range(1, 7):
        runs, wicket = simulate_ball(striker, bowler, df)
        total_runs += runs
        print(f"Ball {ball}: Runs - {runs}, Wicket - {'Yes' if wicket else 'No'}")
        
        if runs % 2 != 0:  # Switch strikers on odd runs
            striker, non_striker = non_striker, striker
        
        if wicket:
            wickets += 1
            break  # Simulation stops if a wicket is taken

    return total_runs, wickets

# Example usage
striker = 'Player A'
non_striker = 'Player B'
bowler = 'Bowler X'
simulate_over(striker, non_striker, bowler, df)
