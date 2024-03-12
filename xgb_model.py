import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from sparse_distribution import simulate_outcome
from ball_record import Ball
import math

batter_list = []
bowler_list = []
batting_stats = {}
bowling_stats = {}



# Load datasets
df_matches = pd.read_csv("match_data.csv")
df_players = pd.read_csv("players.csv")

#merging player statistics into the match data
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
X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(preprocessor.fit_transform(X), y_runs, test_size=0.2, random_state=42)

X_train_wickets, X_test_wickets, y_train_wickets, y_test_wickets = train_test_split(preprocessor.fit_transform(X), y_wickets, test_size=0.2, random_state=42)



# Initialize and train the model
'''model_runs = XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=4, learning_rate=0.025, random_state=42)
model_wickets = XGBRegressor(objective='binary:logistic', n_estimators=300, max_depth=4, learning_rate=0.025, random_state=42)'''


# Hyperparameter tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.2]
# }

# grid_search = GridSearchCV(model_runs, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train_runs, y_train_runs)

# # Print the best hyperparameters
# print("Best hyperparameters:", grid_search.best_params_)


'''model_runs.fit(X_train_runs, y_train_runs)
model_wickets.fit(X_train_wickets, y_train_wickets)



#Calculate performance metrics
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
dump(model_wickets, 'model_wickets.joblib')'''

#Load the models
runs_model = load('model_runs.joblib')
wickets_model = load('model_wickets.joblib')


#Simulates outcome when data is sparse
def predict_using_stats(striker, bowler):
    batsman_stats = df_players[df_players['name'] == striker].iloc[0]
    bowler_stats = df_players[df_players['name'] == bowler].iloc[0]
    predicted_runs = simulate_outcome(striker, bowler)

    # Calculate wicket probability
    batsman_dismissal_rate = batsman_stats['dismissedPerBall']
    bowler_wicket_rate = bowler_stats['wicketsPerBall']
    wicket_probability = (batsman_dismissal_rate + bowler_wicket_rate) / 2
    


    
    return predicted_runs, wicket_probability

# Function to simulate a ball's outcome
def simulate_ball(striker, non_striker, bowler, phase):
    wicket_probability = 0
    if is_data_sparse(striker, bowler, phase):
        predicted_runs, wicket_probability = predict_using_stats(striker, bowler)
    else:
        striker_data = df_players[df_players['name'] == striker]
        # Get the statistics attributes
        runsScoredPerBall = striker_data['runsScoredPerBall'].iloc[0]
        dismissedPerBall = striker_data['dismissedPerBall'].iloc[0]
        bowler_data = df_players[df_players['name'] == bowler]
        wicketsPerBall = bowler_data['wicketsPerBall'].iloc[0]
        runsConcededPerBall = bowler_data['runsConcededPerBall'].iloc[0]
        # Prepare the input for the model
        input_features = pd.DataFrame([[striker, non_striker, bowler, phase, runsScoredPerBall,
                                        dismissedPerBall, wicketsPerBall, runsConcededPerBall]],
                                      columns=['striker', 'non_striker', 'bowler', 'phase',
                                               'runsScoredPerBall', 'dismissedPerBall',
                                               'wicketsPerBall', 'runsConcededPerBall' ])
        input_features_transformed = preprocessor.transform(input_features)
        predicted_runs = runs_model.predict(input_features_transformed)[0]
        wicket_probability = wickets_model.predict(input_features_transformed)[0]
        predicted_runs = np.random.poisson(lam = predicted_runs)
    
    if wicket_probability > np.random.rand():
        is_wicket = True
    else:
        is_wicket = False

    if is_wicket:
        predicted_runs = 0
    else:
        predicted_runs = predicted_runs if predicted_runs != 5  else 6
        predicted_runs = min(predicted_runs, 6)
    

    return predicted_runs, is_wicket

ball_list = []


#batting_stats = {batter: {'runs_scored': 0, 'balls_faced': 0, 'is_out': False} for batter in batter_list}
#bowling_stats = {bowler: {'overs_bowled': 0, 'runs_conceded': 0, 'wickets_taken': 0, 'balls_bowled': 0} for bowler in bowler_list}



def update_batting_stats(batsman, runs, is_wicket, bowler):
    #Add stats to dictionary
    batting_stats[batsman]["runs_scored"] += runs
    batting_stats[batsman]["balls_faced"] += 1
    #Set dismissed by property
    if is_wicket:
        batting_stats[batsman]["is_out"] = True
        batting_stats[batsman]["dismissed_by"] = bowler



def update_bowling_stats(bowler, runs, is_wicket):
    bowling_stats[bowler]['runs_conceded'] += runs
    bowling_stats[bowler]['balls_bowled'] += 1
    if is_wicket:
        bowling_stats[bowler]['wickets_taken'] += 1

def print_statistics():
    print("\nBatting Statistics:")
    #Looping through batters
    for batter, stats in batting_stats.items():
        if stats['is_out']:
            #For batsman who are out
            strike_rate = round((stats['runs_scored']/stats['balls_faced'])*100, 2)
            print(f"{batter}: Runs Scored = {stats['runs_scored']}, Balls Faced = {stats['balls_faced']}, Strike rate = {strike_rate}, "
      f"Dismissed by {stats['dismissed_by']}")
        elif stats['balls_faced'] > 0:
            #Batsman who batted but not out
            strike_rate = round((stats['runs_scored']/stats['balls_faced'])*100, 2)
            print(f"{batter}: Runs Scored = {stats['runs_scored']}, Balls Faced = {stats['balls_faced']}, Strike rate = {strike_rate}")
        else:
            #Never batted
            print(batter + " did not bat")

    print("\nBowling Statistics:")
    for bowler, stats in bowling_stats.items():
        overs, partial_overs = divmod(stats['balls_bowled'], 6)
        #Calculating whole number of overs and then number of balls in partial over
        overs_bowled = f"{overs}.{partial_overs}"
        economy_rate = stats['runs_conceded'] / (stats['balls_bowled'] / 6) if stats['balls_bowled'] > 0 else 0
        print(f"{bowler}: Overs Bowled = {overs_bowled}, Runs Conceded = {stats['runs_conceded']}, "
              f"Wickets Taken = {stats['wickets_taken']}, Economy Rate = {economy_rate:.2f}")


#Define when data is sparse, returns Bool
def is_data_sparse(striker, bowler, phase):
    matches = ((df_matches[(df_matches['striker'] == striker) & (df_matches['bowler'] == bowler)]))
    phase_matches = (df_matches[(df_matches['striker'] == striker) & (df_matches['bowler'] == bowler) & (df_matches['phase']==phase)])
    return len(matches) < 4 or len(phase_matches) < 1



#Calculate phase based on over number
def calculate_phase(over_number):
    return (over_number // 10) + 1



#Simulate entire over, update total statistics
def simulate_over(bowler, striker, non_striker, next_batters, over_number, batter_index, wickets_fallen, total_runs, first_innings_score, innings):
    phase = calculate_phase(over_number)
    new_batter_index = batter_index
    #Simulate each ball in over
    for i in range(1, 7):
        runs, is_wicket = simulate_ball(striker, non_striker, bowler, phase)
        ball_list.append(Ball(over_number, i, striker, bowler, int(runs), is_wicket, innings))
        ball_num = round(over_number + 0.1*i, 1)
        update_batting_stats(striker, runs, is_wicket, bowler)
        update_bowling_stats(bowler, runs, is_wicket)

        #print(f"Striker: {striker}, Bowler: {bowler}, Ball {ball_num}: Phase: {phase}, Runs: {runs}, Wicket: {is_wicket}")

        total_runs += runs
        if runs % 2 != 0:
            striker, non_striker = non_striker, striker
        if innings == 2 and total_runs > first_innings_score:
                break
        #Stopping condition
        if is_wicket:
            if wickets_fallen < 9:
                striker = next_batters[new_batter_index]
                new_batter_index += 1
            wickets_fallen += 1
            if wickets_fallen == 10:
                break
    striker, non_striker = non_striker, striker
    return wickets_fallen, striker, non_striker, new_batter_index, total_runs

def simulate_innings(batter_list1, bowler_list1, innings, first_innings_score):
    global batter_list
    global bowler_list
    global batting_stats
    global bowling_stats
        
    batter_list = batter_list1
    bowler_list = bowler_list1
    
    batting_stats = {batter: {'runs_scored': 0, 'balls_faced': 0, 'is_out': False} for batter in batter_list}
    bowling_stats = {bowler: {'overs_bowled': 0, 'runs_conceded': 0, 'wickets_taken': 0, 'balls_bowled': 0} for bowler in bowler_list}
    
    wickets_fallen = 0
    batter_index = 2 #this was previously set as 1, but caused 2nd batsman to duplicate if first batter got out
    striker, non_striker = batter_list[0], batter_list[1]
    over_number = 0
    total_runs = 0
    
    #While loop, stopping condition ends innings
    while (wickets_fallen < 10 and over_number < 50) and (innings == 1  or (total_runs  <= first_innings_score)):
        bowler = bowler_list[over_number % len(bowler_list)]
        #Over simulation
        wickets_fallen, striker, non_striker, batter_index, total_runs = simulate_over(bowler,
            striker, non_striker, batter_list, over_number, batter_index, wickets_fallen, total_runs, first_innings_score, innings)
        over_number += 1
        if batter_index > len(batter_list):
            break

    print(f"Innings ended with {wickets_fallen} wickets fallen over {over_number} overs and total runs scored: {total_runs}.")
    #print_statistics()
    return ball_list, total_runs


#simulate_innings(batter_list, bowler_list)


        


        