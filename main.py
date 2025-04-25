#!/usr/bin/env python3
import sys
import numpy as np

# Attempt to import plotting libraries
def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg for interactivity
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        return plt, mpimg
    except ImportError:
        sys.stderr.write(
            "Error: matplotlib is required for plotting but is not installed.\n"
            "Please install it via 'pip install matplotlib' or your environment manager.\n"
        )
        sys.exit(1)

plt, mpimg = _import_matplotlib()
plt.rcParams['backend'] = 'TkAgg'

from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from data_loader import load_temperature_data, load_long_data

# ---------- Fit helpers ----------

def poly_fit(degree):
    """Polynomial fit of specified degree."""
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit

def exp_fit(x: np.ndarray, y: np.ndarray):
    """
    Fit y ≈ c * exp(r*(x - x0)) + b, with data-driven initial guess.
    """
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

# ---------- Interactive grid helper ----------

def interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=50, future_end=2050,
                                recent_data=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    annotations, halos, plot_data = [], [], []
    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label='Data', zorder=2)
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

        ann = ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points',
                          bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle='->'))
        ann.set_visible(False)
        halo, = ax.plot([], [], 'o', ms=np.sqrt(scatter_size), mec='yellow',
                        mfc='none', mew=2, zorder=4)

        annotations.append(ann)
        halos.append(halo)
        plot_data.append((ax, x_arr, y_arr, model, last_x))

    def on_move(event):
        for ann, halo in zip(annotations, halos):
            ann.set_visible(False)
            halo.set_data([], [])
        if event.inaxes is None or event.xdata is None:
            fig.canvas.draw_idle()
            return
        for (ax, x_arr, y_arr, model, last_x), ann, halo in zip(plot_data, annotations, halos):
            if event.inaxes != ax:
                continue
            year_hover = event.xdata
            if year_hover > last_x:
                year = int(round(year_hover))
                y_val = model(year)
                obs_str = '-'
                pred_str = f'{y_val:.2f}'
                ann.xy = (year, y_val)
                ann.set_text(f"Year: {year}\nPredicted: {pred_str}")
                halo.set_data([year], [y_val])
            else:
                idx = np.argmin(np.abs(x_arr - year_hover))
                year = int(x_arr[idx])
                y_val = y_arr[idx]
                pred_val = model(year)
                obs_str = f'{y_val:.2f}'
                pred_str = f'{pred_val:.2f}'
                ann.xy = (year, y_val)
                ann.set_text(f"Year: {year}\nObserved: {obs_str}\nPredicted: {pred_str}")
                halo.set_data([year], [y_val])
            ann.set_visible(True)
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()
    return fig, axes

if __name__ == '__main__':
    x_temp, y_temp = load_temperature_data()
    x_long, y_long = load_long_data()

    quadratic_fit_func = poly_fit(2)
    exp_fit_func = exp_fit

    quadratic_model = quadratic_fit_func(x_temp, y_temp)
    exp_model = exp_fit_func(x_temp, y_temp)

    def reuse_quadratic(x_arr, y_arr):
        return quadratic_model

    def reuse_exp(x_arr, y_arr):
        return exp_model

    datasets = [
        (x_temp, y_temp, quadratic_fit_func, 'Quadratic Fit (good recently / bad long-term fit)', '', 'Temperature'),
        (x_temp, y_temp, exp_fit_func, 'Exponential Fit (good recently / good long-term fit)', '', 'Temperature'),
        (x_long, y_long, reuse_quadratic, 'Extrapolation (Quadratic)', 'Year', 'Temperature'),
        (x_long, y_long, reuse_exp, 'Extrapolation (Exponential)', 'Year', 'Temperature'),
    ]

    interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=30, future_end=2050,
                                recent_data=(x_temp, y_temp))
