import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load

# Load the player statistics
players_df = pd.read_csv("players.csv")

# Load the match data
df = pd.read_csv("odi_match_data2.csv")

# Preprocessing the match data
# Only keeping necessary columns
df = df[['venue', 'striker', 'non_striker', 'bowler', 'runs_off_bat', 'is_wicket', 'phase']]

# Encoding categorical variables
categorical_features = ['venue', 'striker', 'non_striker', 'bowler', 'phase']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into features and target labels
X = df.drop(columns=['runs_off_bat', 'is_wicket'])
y_runs = df['runs_off_bat']
y_wicket = df['is_wicket']

# Split the data into training and testing sets
X_train, X_test, y_runs_train, y_runs_test, y_wicket_train, y_wicket_test = train_test_split(
    X, y_runs, y_wicket, test_size=0.2, random_state=42)

# Create the models
model_runs = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
model_wicket = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

# Fit the models
'''model_runs.fit(X_train, y_runs_train)
dump(model_runs, 'runs_model_byphase.joblib')

model_wicket.fit(X_train, y_wicket_train)
dump(model_wicket, 'wicket_model_byphase.joblib')'''

model_runs = load("runs_model_byphase.joblib")
model_wicket = load("wicket_model_byphase.joblib")


# Function to predict using player statistics
def predict_using_stats(striker, bowler):
    batsman_stats = players_df[players_df['name'] == striker].iloc[0]
    bowler_stats = players_df[players_df['name'] == bowler].iloc[0]

    # Calculate weighted runs based on batsman and bowler stats
    batsman_runs_per_ball = batsman_stats['runsScoredPerBall']
    bowler_runs_per_ball = bowler_stats['runsConcededPerBall']
    lambda_runs = (batsman_runs_per_ball + bowler_runs_per_ball) / 2

    # Predict runs using Poisson distribution
    predicted_runs = np.random.poisson(lambda_runs)

    # Calculate wicket probability
    batsman_dismissal_rate = batsman_stats['dismissedPerBall']
    bowler_wicket_rate = bowler_stats['wicketsPerBall']
    wicket_probability = (batsman_dismissal_rate + bowler_wicket_rate) / 2

    return predicted_runs, wicket_probability

# Function to simulate a ball's outcome
def simulate_ball(venue, striker, non_striker, bowler, phase):
    if is_data_sparse(striker, bowler):
        predicted_runs, wicket_probability = predict_using_stats(striker, bowler)
        predicted_wicket = np.random.rand() < wicket_probability
    else:
        # Prepare the input for the model
        input_features = pd.DataFrame([[venue, striker, non_striker, bowler, phase]],
                                      columns=['venue', 'striker', 'non_striker', 'bowler', 'phase'])
        predicted_runs = model_runs.predict(input_features)[0]
        predicted_runs = np.random.poisson(predicted_runs)
        predicted_wicket = model_wicket.predict(input_features)[0]

    # Round predicted_runs and ensure it's not 5
    #predicted_runs = np.round(predicted_runs, 0).astype(int)
    
    
    predicted_runs = predicted_runs if predicted_runs != 5 else 6
    predicted_runs = min(predicted_runs, 6)

    return predicted_runs, (predicted_wicket)


def is_data_sparse(striker, bowler):
    matches = df[(df['striker'] == striker) & (df['bowler'] == bowler)]
    return len(matches) < 4

# Example usage:
# Simulate the outcome of a ball
venue = 'Holkar Cricket Stadium, Indore'
striker = 'V Kohli'
non_striker = 'Shubman Gill'
bowler = 'JR Hazlewood'
phase = 0
total = 0
wickets = 0
for i in range(1,6):
    phase += 1
    runs, wicket = simulate_ball(venue, striker, non_striker, bowler, phase)
    total += runs
    print(f"Ball {i}: Predicted runs: {runs}, Predicted wicket: { wicket}")
    wickets += wicket

print(str(total), str(wickets))