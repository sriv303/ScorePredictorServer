import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

match_data = pd.read_csv("C:/Users/Abhi/Documents/Schoolwork/Computer Science/CricketPredictor/Database/odi_match_data.csv", low_memory=False)

match_data.describe()