import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv("Temperature_Data.csv")

# Convert the 'year' and 'temp_anomaly' columns to numpy arrays
x = df['year'].to_numpy()
y = df['temp_anomaly'].to_numpy()

# Optional: Print the arrays to verify the results
print("x (years):", x)
print("y (temp_anomaly):", y)
