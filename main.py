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


def exponential(x_arr, c, r, b=0):
    """Exponential model: c * exp(r * x) + b."""
    return c * np.exp(r * x_arr) + b


def general_curve_fit(model_func, p0=None, maxfev=2000):
    """
    Generic wrapper for scipy.optimize.curve_fit.
    model_func must have signature model_func(x_arr, *params).
    p0 provides initial guess for params; if a tuple contains None,
    it's replaced by the first y-value of the fit call.
    Returns a function fit(x_arr, y_arr).
    """
    def fit(x_arr, y_arr):
        # build actual initial guess
        p0_actual = None
        if p0 is not None:
            if isinstance(p0, (list, tuple)):
                p0_list = []
                for val in p0:
                    if val is None:
                        p0_list.append(y_arr[0])
                    else:
                        p0_list.append(val)
                p0_actual = tuple(p0_list)
            else:
                p0_actual = p0
        params, _ = curve_fit(model_func, x_arr, y_arr, p0=p0_actual, maxfev=maxfev)
        return lambda x_new: model_func(x_new, *params)
    return fit


def loess_fit(x_arr, y_arr, frac=0.3):
    """LOESS fit using statsmodels.lowess."""
    loess_res = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    xs, ys = loess_res[:, 0], loess_res[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)

# ---------- Interactive grid helper ----------

def interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=50, future_end=2050):
    """
    Display a grid of interactive regression plots.

    On hover, shows:
      - Year (discrete integer)
      - Observed (or '-')
      - Predicted (curve value)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    annotations, halos, plot_data = [], [], []
    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label='Data', zorder=2)
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
            if event.inaxes == ax:
                year = int(round(event.xdata))
                year = max(min(year, int(future_end)), int(np.min(x_arr)))
                if year > last_x:
                    obs_str = '-'
                    y_val = model(year)
                else:
                    idx = np.where(x_arr == year)[0]
                    if idx.size:
                        obs_str = f"{y_arr[idx[0]]:.2f}"
                        y_val = y_arr[idx[0]]
                    else:
                        obs_str = '-'
                        y_val = model(year)
                pred = model(year)
                ann.xy = (year, y_val)
                ann.set_text(f"Year: {year}\nObserved: {obs_str}\nPredicted: {pred:.2f}")
                halo.set_data([year], [y_val])
                ann.set_visible(True)
                fig.canvas.draw_idle()
                return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()
    return fig, axes

if __name__ == '__main__':
    # Load data
    x_temp, y_temp = load_temperature_data()
    x_long, y_long = load_long_data()

    # Create fit functions
    linear_fit_func = poly_fit(1)
    exp_fit_func    = general_curve_fit(exponential, p0=(None, 0.01, 0))

    # Fit models on temperature data
    linear_model = linear_fit_func(x_temp, y_temp)
    exp_model    = exp_fit_func(x_temp, y_temp)

    # Wrappers to reuse fitted parameters on any dataset
    def reuse_linear(x_arr, y_arr):
        return linear_model
    def reuse_exp(x_arr, y_arr):
        return exp_model

    # Prepare the four interactive plots
    datasets = [
        (x_temp, y_temp, linear_fit_func,          'Linear Fit (Temp)',             'Year', 'Value'),
        (x_temp, y_temp, exp_fit_func,             'Exponential Fit (Temp)',        'Year', 'Value'),
        (x_long, y_long, reuse_linear,             'Linear Extrapolation (Long)',   'Year', 'Value'),
        (x_long, y_long, reuse_exp,                'Exponential Extrapolation (Long)','Year', 'Value'),
    ]

    # Display them in a 2×2 interactive grid
    interactive_regression_grid(datasets, nrows=2, ncols=2, scatter_size=50, future_end=2050)
