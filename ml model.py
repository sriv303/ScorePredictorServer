from datetime import datetime

# Get the current time
now = datetime.now()

# Format the current time as a string (e.g., HH:MM:SS)
current_time = now.strftime("%H:%M:%S")

# Print the current time
print("Current time:", current_time)


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
    ('regressor', RandomForestRegressor(random_state=42))
])

pipeline_wicket = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
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
dump(pipeline_wicket, 'pipeline_wicket.joblib')
