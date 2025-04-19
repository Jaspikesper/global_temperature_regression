import os
import csv

# Define file paths
input_path = r"C:\Users\jaspe\PycharmProjects\PythonProject5\data\ensemble.txt"
output_path = r"C:\Users\jaspe\PycharmProjects\PythonProject5\data\long.csv"

# Step 0: Read the data
with open(input_path, 'r') as f:
    lines = f.readlines()

# Process the lines
year_data = {}
for line in lines:
    parts = line.strip().split()
    if not parts:
        continue

    try:
        year = int(parts[0])
    except ValueError:
        continue  # skip lines that don't start with a year

    if 1000 <= year < 1961:
        observations = list(map(float, parts[1:]))
        if observations:
            year_data[year] = sum(observations) / len(observations)

# Step 2 & 3: Write to CSV
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Year', 'AverageTemperature'])
    for year in sorted(year_data):
        writer.writerow([year, round(year_data[year], 4)])

print(f"Data written to {output_path}")
