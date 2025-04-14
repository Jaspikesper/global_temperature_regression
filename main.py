import pandas as pd
import numpy as np
import single_variable_regression
from single_variable_regression import *
from regression_plot import *

if __name__ == "__main__":

    data = pd.read_csv('merged_co2_temp.csv')
    x = data['mean'].values
    y = data['temp_anomaly'].values

    n = len(x)
    split = 0.8
    N = int(split*n)
    x_fit = x[:N]
    x_test = x[N:]
    y_fit = y[:N]
    y_test = y[N:]
    x_test = np.hstack([x_fit[0], x_test])
    y_test = np.hstack([y_fit[0], y_test])

    mhat, bhat, r, r2 = linear_regression(x_fit, y_fit) # <- This could be any regression function
    single_variable_regression.predict(x_fit, y_fit, mhat, bhat)
    fig, ax = regression_plot(x_test, y_test, mhat, bhat, r, r2)
    ax.set_xlim(x[0])
    ax.set_ylim(0)
    plt.show()
    print('done!')