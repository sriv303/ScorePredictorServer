import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from sparse_distribution import simulate_outcome


# Load datasets
df_matches = pd.read_csv("odi_match_data2.csv")
df_players = pd.read_csv("players.csv")

# Example of merging player statistics into the match data
# Assuming 'striker' and 'bowler' columns in df_matches match 'name' in df_players
df_matches = df_matches.merge(df_players[['name', 'runsScoredPerBall', 'dismissedPerBall']], left_on='striker', right_on='name', how='left')
df_matches = df_matches.merge(df_players[['name', 'wicketsPerBall', 'runsConcededPerBall']], left_on='bowler', right_on='name', how='left')

# Select features and target
X = df_matches[[ 'striker', 'non_striker', 'bowler', 'runsScoredPerBall', 'dismissedPerBall', 'wicketsPerBall', 'runsConcededPerBall', 'phase']]
y_runs = df_matches['runs_off_bat']
y_wickets = df_matches['is_wicket'].astype(int)

# Encoding categorical variables
categorical_features = [ 'striker', 'non_striker', 'bowler', 'phase']
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)



# Split the data
X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_processed, y_runs, test_size=0.2, random_state=42)

X_train_wickets, X_test_wickets, y_train_wickets, y_test_wickets = train_test_split(preprocessor.transform(X), y_wickets, test_size=0.2, random_state=42)



# Initialize and train the model
model_runs = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
model_wickets = XGBRegressor(objective='binary:logistic', n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)


# Hyperparameter tuning (simplified example)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(model_runs, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_runs, y_train_runs)

# Best model
best_model = grid_search.best_estimator_

#best params were found to be n_estimators = 100, max_depth = 5, lr = 0.05

model_runs.fit(X_train_runs, y_train_runs)
model_wickets.fit(X_train_wickets, y_train_wickets)
# Predictions
y_pred_runs = model_runs.predict(X_test_runs)
mse_runs = mean_squared_error(y_test_runs, y_pred_runs)
r2_runs = r2_score(y_test_runs, y_pred_runs)

y_pred_wickets = model_wickets.predict(X_test_wickets)
# Since model_wickets predicts probabilities, threshold to classify wickets
y_pred_wickets_binary = (y_pred_wickets >= 0.5).astype(int)
accuracy_wickets = accuracy_score(y_test_wickets, y_pred_wickets_binary)

print(f"Runs Model - MSE: {mse_runs}, R^2: {r2_runs}")
print(f"Wickets Model - Accuracy: {accuracy_wickets}")

# Save the models
dump(model_runs, 'model_runs.joblib')
dump(model_wickets, 'model_wickets.joblib')


runs_model = load('model_runs.joblib')
wickets_model = load('model_wickets.joblib')


def predict_using_stats(striker, bowler):
    batsman_stats = df_players[df_players['name'] == striker].iloc[0]
    bowler_stats = df_players[df_players['name'] == bowler].iloc[0]

    '''# Calculate weighted runs based on batsman and bowler stats
    batsman_runs_per_ball = batsman_stats['runsScoredPerBall']
    bowler_runs_per_ball = bowler_stats['runsConcededPerBall']
    lambda_runs = (batsman_runs_per_ball + bowler_runs_per_ball) / 2

    # Predict runs using Poisson distribution
    predicted_runs = np.random.poisson(lambda_runs)'''
    
    predicted_runs = simulate_outcome(striker, bowler)

    # Calculate wicket probability
    batsman_dismissal_rate = batsman_stats['dismissedPerBall']
    bowler_wicket_rate = bowler_stats['wicketsPerBall']
    if batsman_stats['isBatsman'] != 1:
        wicket_probability = (batsman_dismissal_rate + bowler_wicket_rate) / 1.25
    else:
        wicket_probability = (batsman_dismissal_rate + bowler_wicket_rate) / 2


    
    return predicted_runs, wicket_probability

# Function to simulate a ball's outcome
def simulate_ball(striker, non_striker, bowler, phase):
    if is_data_sparse(striker, bowler, phase):
        predicted_runs, wicket_probability = predict_using_stats(striker, bowler)
        #predicted_wicket = np.random.rand() < wicket_probability
        predicted_wicket = wicket_probability
    else:
        striker_data = df_players[df_players['name'] == striker]
        # Get the 'runsScoredPerBall' attribute
        runsScoredPerBall = striker_data['runsScoredPerBall'].iloc[0]
        dismissedPerBall = striker_data['dismissedPerBall'].iloc[0]
        bowler_data = df_players[df_players['name'] == bowler]
        wicketsPerBall = bowler_data['wicketsPerBall'].iloc[0]
        runsConcededPerBall = bowler_data['runsConcededPerBall'].iloc[0]
        # Prepare the input for the model
        input_features = pd.DataFrame([[striker, non_striker, bowler, phase, runsScoredPerBall, dismissedPerBall, wicketsPerBall, runsConcededPerBall]],
                                      columns=['striker', 'non_striker', 'bowler', 'phase', 'runsScoredPerBall', 'dismissedPerBall', 'wicketsPerBall', 'runsConcededPerBall' ])
        input_features_transformed = preprocessor.transform(input_features)
        predicted_runs = runs_model.predict(input_features_transformed)[0]
        predicted_wicket = wickets_model.predict(input_features_transformed)[0]
    
    
    predicted_runs = predicted_runs if predicted_runs != 5 else 6
    predicted_runs = min(predicted_runs, 6)

    return predicted_runs, (predicted_wicket)


def is_data_sparse(striker, bowler, phase):
    matches = ((df_matches[(df_matches['striker'] == striker) & (df_matches['bowler'] == bowler)]))
    phase_matches = (df_matches[(df_matches['striker'] == striker) & (df_matches['bowler'] == bowler) & (df_matches['phase']==phase)])
    return len(matches) < 4 or len(phase_matches) < 1


striker = 'RG Sharma'
non_striker = 'V Kohli'
bowler = 'JR Hazlewood'
phase = 1
total = 0
wickets = 0
for i in range(1, 7):
    runs, wicket = simulate_ball(striker, non_striker, bowler, phase)
    total += runs
    print(f"Ball {i}: Predicted runs: {runs}, Predicted wicket: {wicket}")
    wickets += wicket

print(str(total), str(wickets))
