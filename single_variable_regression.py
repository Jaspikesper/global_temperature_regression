import numpy as np
import pandas as pd

data = pd.read_csv('merged_co2_temp.csv')

x = data['mean'].values
y = data['temp_anomaly'].values

def linear_regression(xvar, yvar):

    mhat = np.cov(xvar, yvar)[0, 1] / np.var(xvar) # Regression slope
    bhat = np.mean(yvar) - mhat * np.mean(xvar) # Intercept

    r = np.corrcoef(xvar, yvar)[0, 1]
    r2 = r**2

    return mhat, bhat, r, r2

def predict():
    print("not implemented yet")