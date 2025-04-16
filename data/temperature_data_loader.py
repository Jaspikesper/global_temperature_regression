import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv("data/Temperature_Data.csv")

# Convert the 'year' and 'temp_anomaly' columns to numpy arrays
x = df['year'].to_numpy()
y = df['temp_anomaly'].to_numpy()


if __name__ == '__main__':
    print("x (years):", x)
    print("y (temp_anomaly):", y)
