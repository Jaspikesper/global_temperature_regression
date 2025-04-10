import numpy as np
from single_regression_plot import regression_plot
import pandas as pd

data = pd.read_csv('merged_co2_temp.csv')

x = data['mean'].values
y = data['temp_anomaly'].values

mhat = np.cov(x, y)[0, 1] / np.var(x)
bhat = np.mean(y) - mhat * np.mean(x)

regression_plot(x, y)

print(mhat)
print(bhat)