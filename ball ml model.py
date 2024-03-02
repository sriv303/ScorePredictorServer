import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from joblib import dump, load
from sklearn.metrics import mean_squared_error, r2_score

# Load the player statistics
players_df = pd.read_csv("players.csv")

# Load the match data
df = pd.read_csv("odi_match_data2.csv")

# Preprocessing the match data
# Only keeping necessary columns
df = df[['venue', 'striker', 'non_striker', 'bowler', 'runs_off_bat', 'is_wicket', 'ball']]

# Encoding categorical variables
categorical_features_runs = [ 'striker', 'bowler', 'ball']
categorical_features_wickets = [ 'striker', 'bowler']


categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_runs = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features_runs)
    ])

preprocessor_wicket = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features_wickets)
    ])
# Split the data into features and target labels
X = df.drop(columns=['runs_off_bat', 'is_wicket'])
y_runs = df['runs_off_bat']
y_wicket = df['is_wicket']

# Split the data into training and testing sets
X_train, X_test, y_runs_train, y_runs_test, y_wicket_train, y_wicket_test = train_test_split(
    X, y_runs, y_wicket, test_size=0.2, random_state=42)

# Create the models
model_runs = Pipeline(steps=[('preprocessor', preprocessor_runs),
                             ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
model_wicket = Pipeline(steps=[('preprocessor', preprocessor_wicket),
                               ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

# Fit the models

'''model_runs.fit(X_train, y_runs_train)
dump(model_runs, 'runs_model_byball.joblib')

model_wicket.fit(X_train, y_wicket_train)
dump(model_wicket, 'wicket_model_byball.joblib')'''

model_runs = load("runs_model_byball.joblib")
model_wicket = load("wicket_model_byball.joblib")

y_runs_pred = model_runs.predict(X_test)
mse_runs = mean_squared_error(y_runs_test, y_runs_pred)
r2_runs = r2_score(y_runs_test, y_runs_pred)

print(mse_runs, r2_runs)
 


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
def simulate_ball(striker, non_striker, bowler, ball):
    if is_data_sparse(striker, bowler, ball):
        predicted_runs, wicket_probability = predict_using_stats(striker, bowler)
        #predicted_wicket = np.random.rand() < wicket_probability
        predicted_wicket = wicket_probability
    else:
        # Prepare the input for the model
        input_features_runs = pd.DataFrame([[striker, bowler, ball]],
                                      columns=['striker', 'bowler', 'ball'])
        input_features_wicket = pd.DataFrame([[striker, bowler]],
                                      columns=['striker','bowler'])
        predicted_runs = model_runs.predict(input_features_runs)[0]
        predicted_wicket = model_wicket.predict(input_features_wicket)[0]

    # Round predicted_runs and ensure it's not 5
    #predicted_runs = np.round(predicted_runs, 0).astype(int)
    
    
    predicted_runs = predicted_runs if predicted_runs != 5 else 6
    predicted_runs = min(predicted_runs, 6)

    return predicted_runs, (predicted_wicket)


def is_data_sparse(striker, bowler, ball):
    matches = ((df[(df['striker'] == striker) & (df['bowler'] == bowler)]))
    ball_matches = (df[(df['striker'] == striker) & (df['bowler'] == bowler) & (df['ball']==ball)])
    return len(matches) < 4 or len(ball_matches) < 1

# Example usage:
# Simulate the outcome of a ball
venue = 'Holkar Cricket Stadium, Indore'
striker = 'RG Sharma'
non_striker = 'RG Sharma'
bowler = 'A Zampa'
ball = 8.0
total = 0
wickets = 0
for i in range(1, 7):
    ball += 0.1
    ball.__round__(1)
    runs, wicket = simulate_ball(striker, non_striker, bowler, ball)
    total += runs
    print(f"Ball {i}: Predicted runs: {runs}, Predicted wicket: {wicket}")
    wickets += wicket

print(str(total), str(wickets))