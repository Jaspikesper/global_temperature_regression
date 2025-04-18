# subplot.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from data.temperature_data_loader import x, y

# Ensure the TkAgg backend for interactivity
plt.rcParams['backend'] = 'TkAgg'

# ---------- Fit helpers ----------

def poly_fit(degree):
    """Return a fit(x, y) for a polynomial of given degree."""
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exponential_fit(x_arr, y_arr):
    """Fit y = a * exp(b * (x - x0)) to stabilize scaling."""
    # Center x to avoid huge exponentials on large year values
    x0 = np.min(x_arr)
    def model(x_in, a, b):
        return a * np.exp(b * (x_in - x0))
    # Initial guess: a ~ first y, b small positive
    p0 = (y_arr[0] if len(y_arr)>0 else 1.0, 0.01)
    params, _ = curve_fit(model, x_arr, y_arr, p0=p0, maxfev=2000)
    return lambda x_new: model(x_new, *params)


def loess_fit(x_arr, y_arr, frac=0.3):
    """Return a LOESS fit function."""
    loess_res = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    return lambda x_new: np.interp(x_new, loess_res[:, 0], loess_res[:, 1])

# ---------- Interactive grid helper ----------
def interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=50, future_end=2050):
    """
    Plot multiple regression results interactively.

    Parameters
    ----------
    datasets : list of tuples
        Each tuple is (x_array, y_array, fit_func, title, xlabel, ylabel).
    nrows, ncols : int
        Grid dimensions (max 4 panels).
    scatter_size : int
        Marker size for data.
    future_end : float
        Extends fit to this x-value if max(x)<future_end.

    Returns
    -------
    fig, axes : matplotlib objects
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    annotations = []
    halos = []
    plot_data = []

    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        # Plot data + fit line
        ax.scatter(x_arr, y_arr, s=scatter_size, label="Data", zorder=2)
        ax.plot(x_arr, y_pred, label="Fit", zorder=3)

        # Future projection
        if last_x < future_end:
            fx = np.linspace(last_x, future_end, 100)
            fy = model(fx)
            ax.plot(fx, fy, linestyle='--', label="Future", zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)

        # Hover annotation + halo
        ann = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->")
        )
        ann.set_visible(False)
        annotations.append(ann)

        halo, = ax.plot([], [], 'o',
                        ms=np.sqrt(scatter_size),
                        mec='yellow', mfc='none', mew=2,
                        zorder=4)
        halos.append(halo)

        plot_data.append((x_arr, y_arr, model))

    def on_move(event):
        if event.xdata is None or event.inaxes is None:
            for ann, halo in zip(annotations, halos):
                ann.set_visible(False)
                halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        for ax, (x_arr, y_arr, model), ann, halo in zip(
                axes, plot_data, annotations, halos):
            if event.inaxes == ax:
                x_val = event.xdata
                idx = int(np.argmin(np.abs(x_arr - x_val)))
                snap_x = x_arr[idx]
                snap_y = y_arr[idx]
                cont_y = model(x_val)

                halo.set_data([snap_x], [snap_y])
                ann.xy = (snap_x, snap_y)
                ann.set_text(f"Year = {snap_x}\nValue = {cont_y:.2f}")
                ann.set_visible(True)
                fig.canvas.draw_idle()
                return

        for ann, halo in zip(annotations, halos):
            ann.set_visible(False)
            halo.set_data([], [])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()
    return fig, axes

# ---------- Method map ----------
_method_map = {
    "linear":     (poly_fit(1),      "Linear Fit"),
    "quadratic":  (poly_fit(2),      "Quadratic Fit"),
    "cubic":      (poly_fit(3),      "Cubic Fit"),
    "quartic":    (poly_fit(4),      "Quartic Fit"),
    "exponential":(exponential_fit,   "Exponential Fit"),
    "loess":      (loess_fit,        "LOESS Fit"),
}

# ---------- General regression function ----------
def plot_regression_models(x_arr, y_arr, methods,
                           scatter_size=50, future_end=2050):
    """
    Plot up to 4 regression models interactively.

    Parameters
    ----------
    x_arr, y_arr : array-like
        Data points loaded from `data.temperature_data_loader`.
    methods : list of str
        Keywords among: "linear", "quadratic", "cubic",
        "quartic", "loess", "exponential".
    scatter_size : int
    future_end : float

    Returns
    -------
    fig, axes : matplotlib objects
    """
    if not (1 <= len(methods) <= 4):
        raise ValueError("methods must contain 1 to 4 keywords")

    datasets = []
    for m in methods:
        if m not in _method_map:
            raise KeyError(f"Unknown method '{m}'")
        fit_func, title = _method_map[m]
        datasets.append((x_arr, y_arr, fit_func, title, "Year", "Value"))

    n = len(datasets)
    nrows, ncols = (1, 1) if n == 1 else (1, 2) if n == 2 else (2, 2)

    return interactive_regression_grid(
        datasets,
        nrows=nrows,
        ncols=ncols,
        scatter_size=scatter_size,
        future_end=future_end
    )

# ---------- Demo ----------
if __name__ == "__main__":
    # Use loaded data x, y from data.temperature_data_loader
    plot_regression_models(
        x, y,
        methods=["linear", "exponential", "loess", "cubic"],
        scatter_size=50,
        future_end=2050
    )