#!/usr/bin/env python3
"""
main.py – interactive 2×2 grid comparing regression fits.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"

from interactivity import attach_hover
from models import poly_fit, exp_fit, REGISTRY                      # NEW
from data_loader import load_temperature_data, load_long_data


# --------------------------- plot helper ----------------------------- #
def interactive_regression_grid(datasets, *, nrows=2, ncols=2,
                                scatter_size=50, future_end=2050,
                                recent_data=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (x_arr, y_arr, fit_func, title) in zip(axes, datasets):
        model   = fit_func(x_arr, y_arr)
        y_pred  = model(x_arr)
        last_x  = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label='Data', zorder=2)
        ax.plot(x_arr, y_pred,             label='Fit',    zorder=3)
        if last_x < future_end:
            fx = np.arange(last_x, future_end + 1)
            ax.plot(fx, model(fx), '--',   label='Future', zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title)
        ax.legend()
        attach_hover(ax, x_arr, y_arr, model,
                     scatter_size=scatter_size,
                     start=int(x_arr.min()), end=future_end)

    plt.tight_layout(); plt.show()


# ---------------------------- driver --------------------------------- #
if __name__ == "__main__":
    x_t, y_t = load_temperature_data()
    x_l, y_l = load_long_data()

    quad = poly_fit(2)
    exp  = exp_fit

    quad_model = quad(x_t, y_t)
    exp_model  = exp(x_t, y_t)

    datasets = [
        (x_t, y_t, quad, 'Quadratic Fit – recent good / long bad'),
        (x_t, y_t, exp,  'Exponential Fit – recent good / long good'),
        (x_l, y_l, lambda *_: quad_model, 'Extrapolation – Quadratic'),
        (x_l, y_l, lambda *_: exp_model,  'Extrapolation – Exponential'),
    ]

    interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=30, future_end=2050,
                                recent_data=(x_t, y_t))
