#!/usr/bin/env python3
"""
main.py
-------
Two-by-two interactive grid comparing regression fits.
"""

from __future__ import annotations

import sys
import numpy as np

# --- matplotlib bootstrap ----------------------------------------------
def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.stderr.write(
            "Error: matplotlib is required for plotting but is not installed.\n"
            "Please install it via 'pip install matplotlib'.\n"
        )
        sys.exit(1)


plt = _import_matplotlib()
plt.rcParams["backend"] = "TkAgg"

from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from interactivity import attach_hover
from data_loader import load_temperature_data, load_long_data


# ---------- Fit helpers -------------------------------------------------
def poly_fit(degree):
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exp_fit(x: np.ndarray, y: np.ndarray):
    x0 = x[0]

    def _model(x_vals, c, r, b):
        return c * np.exp(r * (x_vals - x0)) + b

    b0 = float(np.min(y))
    c0 = float(y[0] - b0)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=5000)
        return lambda x_new: _model(x_new, *params)
    except Exception as exc:
        print(f"[exp_fit] Fit failed with p0={p0}: {exc}")
        mean_y = float(np.mean(y))
        return lambda x_new: np.full_like(x_new, mean_y, dtype=float)


def loess_fit(x_arr, y_arr, frac=0.3):
    loess_res = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    xs, ys = loess_res[:, 0], loess_res[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)


# ---------- Interactive grid helper ------------------------------------
def interactive_regression_grid(
    datasets,
    nrows=2,
    ncols=2,
    scatter_size=50,
    future_end=2050,
    recent_data=None,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label="Data", zorder=2)
        if recent_data and 'Extrapolation' in title:
            x_recent, y_recent = recent_data
            ax.scatter(x_recent, y_recent, s=scatter_size, label='Recent Data', zorder=2)

        ax.plot(x_arr, y_pred, label='Fit', zorder=3)
        if last_x < future_end:
            fx = np.arange(last_x, int(future_end) + 1)
            ax.plot(fx, model(fx), linestyle='--', label='Future', zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # Re-use consolidated hover logic
        attach_hover(
            ax,
            x_arr,
            y_arr,
            model,
            scatter_size=scatter_size,
            start=int(x_arr.min()),
            end=int(future_end),
        )

    plt.tight_layout()
    plt.show()
    return fig, axes


# -------------------------- CLI entry -----------------------------------
if __name__ == '__main__':
    x_temp, y_temp = load_temperature_data()
    x_long, y_long = load_long_data()

    quadratic_fit_func = poly_fit(2)
    quad_model = quadratic_fit_func(x_temp, y_temp)

    exp_fit_func = exp_fit
    exp_model = exp_fit_func(x_temp, y_temp)

    datasets = [
        (x_temp, y_temp, quadratic_fit_func, 'Quadratic Fit (good recently / bad long-term fit)', '', 'Temperature'),
        (x_temp, y_temp, exp_fit_func, 'Exponential Fit (good recently / good long-term fit)', '', 'Temperature'),
        (x_long, y_long, lambda *_: quad_model, 'Extrapolation (Quadratic)', 'Year', 'Temperature'),
        (x_long, y_long, lambda *_: exp_model, 'Extrapolation (Exponential)', 'Year', 'Temperature'),
    ]

    interactive_regression_grid(
        datasets,
        nrows=2,
        ncols=2,
        scatter_size=30,
        future_end=2050,
        recent_data=(x_temp, y_temp),
    )
