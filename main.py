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
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from data_loader import x, y

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
    p0 provides initial guess for params.
    Returns a function fit(x_arr, y_arr).
    """
    def fit(x_arr, y_arr):
        params, _ = curve_fit(model_func, x_arr, y_arr, p0=p0, maxfev=maxfev)
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

# ---------- Method map ----------
_method_map = {
    'linear':     (poly_fit(1), 'Linear'),
    'quadratic':  (poly_fit(2), 'Quadratic'),
    'cubic':      (poly_fit(3), 'Cubic'),
    'quartic':    (poly_fit(4), 'Quartic'),
    'exponential':(general_curve_fit(exponential, p0=(y[0], 0.01, 0)), 'Exponential'),
    'loess':      (loess_fit, 'LOESS'),
}

if __name__ == '__main__':
    methods = ['linear', 'exponential', 'loess', 'cubic']
    datasets = [(x, y, _method_map[m][0], m, 'Year', 'Value') for m in methods]
    interactive_regression_grid(datasets, nrows=2, ncols=2, scatter_size=50, future_end=2050)