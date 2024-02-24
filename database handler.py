import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'odi_match_data.csv'
# Replace 'column_to_delete' with the name of the column you want to delete
columns_to_delete = ['match_id', 'season', 'start_date', 'innings', 'bowling_team', 'batting_team']

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Step 2: Delete the specified columns
df.drop(columns=columns_to_delete, axis=1, inplace=True)

# Step 3: Save the modified DataFrame back to a CSV file
# Replace 'modified_file.csv' with the desired path for the modified CSV
df.to_csv('modified_file.csv', index=False)
