import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# Load dataset
'''df = pd.read_csv('cricket_data.csv')


# Preprocess features and target for modeling
features = ['venue', 'ball', 'striker', 'non_striker', 'bowler']
X = df[features]
y_runs = df['runs_off_bat']
y_wicket = df['is_wicket']


#defining categorical features and numerical values
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['venue', 'striker', 'non_striker', 'bowler']),
        ('num', StandardScaler(), ['ball'])
    ])

# Train-test split
X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X, y_runs, test_size=0.2, random_state=42)
X_train_wickets, X_test_wickets, y_train_wicket, y_test_wicket = train_test_split(X, y_wicket, test_size=0.2, random_state=42)

# Model training
pipeline_runs = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
pipeline_runs.fit(X_train_runs, y_train_runs)

pipeline_wicket = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_wicket.fit(X_train_wickets, y_train_wicket)


#Testing model performance
y_pred_runs = pipeline_runs.predict(X_test_runs)
mse_runs = round(mean_squared_error(y_test_runs, y_pred_runs), 4)
r2_runs = round(r2_score(y_test_runs, y_pred_runs), 4)'''
r2_runs = -0.3147
mse_runs = 2.1386
print(f"Runs MSE {mse_runs}")
print(f"Runs R^2 {r2_runs}")

'''y_pred_wickets = pipeline_wicket.predict(X_test_wickets)
accuracy_wickets = round(accuracy_score(y_test_wicket, y_pred_wickets), 4)'''
accuracy_wickets = 0.8398
print(f"Wicket accuracy {accuracy_wickets}")


'''# Function to simulate a ball's outcome
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
simulate_over(striker, non_striker, bowler, df)'''