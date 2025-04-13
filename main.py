import pandas as pd

import single_variable_regression
from single_variable_regression import *
from regression_plot import *

if __name__ == "__main__":
    # demo with your CSV

    data = pd.read_csv('merged_co2_temp.csv')
    x = data['mean'].values
    y = data['temp_anomaly'].values

    mhat, bhat, r, r2 = linear_regression(x, y) # <- This could be any regression function
    single_variable_regression.predict()
    regression_plot(x, y, mhat, bhat, r, r2)
    print('done!')