import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from data.temperature_data_loader import x, y
plt.rcParams['backend'] = 'TkAgg'


def plot_interactive_regression(
        x, y, fit_func, background_img_path=None,
        title="Interactive Regression Plot",
        xlabel="Year", ylabel="y",
        scatter_label="Data", regression_label="Regression Fit",
        scatter_kwargs=None, line_kwargs=None,
        scatter_size=50, future_end=2050):

    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}

    # ---------- Fit model ----------
    model = fit_func(x, y)
    y_pred = model(x)

    last_x = np.max(x)
    if last_x < future_end:
        future_x = np.linspace(last_x, future_end, 100)
        future_y = model(future_x)
    else:
        future_x, future_y = None, None

    # ---------- Figure / Axes ----------
    fig, ax = plt.subplots()
    ax.set_zorder(1)
    ax.patch.set_alpha(0)

    # ---------- Background image ----------
    if background_img_path:
        try:
            bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
            bg_ax.imshow(mpimg.imread(background_img_path), aspect='auto')
            bg_ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load background image: {e}")

    # ---------- Plot ----------
    ax.scatter(x, y, s=scatter_size, c='black',
               label=scatter_label, zorder=2, **scatter_kwargs)
    ax.plot(x, y_pred, color='orange', linewidth=2,
            label=regression_label, zorder=3, **line_kwargs)

    if future_x is not None:
        ax.plot(future_x, future_y, color='orange', linestyle='dashed',
                linewidth=2, label="Future Prediction", zorder=3, **line_kwargs)
        ax.axvline(last_x, color='orange', linestyle='dotted', linewidth=1)

    # ---------- Labels / legend ----------
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)

    # ---------- Hover ----------
    hover_annotation = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"))
    hover_annotation.set_visible(False)

    halo, = ax.plot([], [], 'o', ms=np.sqrt(scatter_size),
                    mec='yellow', mfc='none', mew=2, zorder=4)

    x_to_idx = {val: idx for idx, val in enumerate(x)}
    current_x = current_y = None
    last_hist_x = last_x

    def on_move(event):
        nonlocal current_x, current_y
        if event.inaxes != ax or event.xdata is None:
            if hover_annotation.get_visible():
                hover_annotation.set_visible(False)
                halo.set_data([], [])
                fig.canvas.draw_idle()
            return

        x_val = event.xdata

        if x_val <= last_hist_x:  # historical
            hovered_x = int(round(x_val))
            if hovered_x in x_to_idx:
                idx = x_to_idx[hovered_x]
                true_x, true_y = x[idx], y[idx]
                pred_y = y_pred[idx]
                if (true_x != current_x) or (true_y != current_y):
                    current_x, current_y = true_x, true_y
                    halo.set_data([true_x], [true_y])
                    hover_annotation.xy = (true_x, pred_y)
                    hover_annotation.set_text(
                        f"Year = {true_x}\ny = {pred_y:.2f}")
                    hover_annotation.set_visible(True)
                    fig.canvas.draw_idle()
            elif hover_annotation.get_visible():
                current_x = current_y = None
                hover_annotation.set_visible(False)
                halo.set_data([], [])
                fig.canvas.draw_idle()
        else:                      # future
            hovered_x = int(round(x_val))                 # <-- snap to integer
            if hovered_x > future_end:
                hovered_x = future_end
            pred_y = model(hovered_x)
            if (hovered_x != current_x) or (pred_y != current_y):
                current_x, current_y = hovered_x, pred_y
                halo.set_data([hovered_x], [pred_y])
                hover_annotation.xy = (hovered_x, pred_y)
                hover_annotation.set_text(
                    f"Year = {hovered_x}\ny = {pred_y:.2f}")
                hover_annotation.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()
    return fig, ax, model


# ---------- Fit helpers ----------
def general_curve_fit(model_func, p0=None, maxfev=2000):
    def fit(x, y):
        params, _ = curve_fit(model_func, x, y, p0=p0, maxfev=maxfev)
        return lambda x_new: model_func(x_new, *params)
    return fit


def loess_fit(x, y, frac=0.3):
    loess_result = lowess(y, x, frac=frac, return_sorted=True)
    return lambda x_new: np.interp(x_new, loess_result[:, 0], loess_result[:, 1])


# ---------- Demo ----------
if __name__ == '__main__':
    np.random.seed(0)
    fit_func = lambda x, y: loess_fit(x, y, frac=0.3)

    plot_interactive_regression(
        x, y, fit_func,
        background_img_path='static/useMe.png',
        title="Interactive Regression Plot",
        xlabel="Year", ylabel="Temperature",
        scatter_label="Temperature", regression_label="Curve Fit",
        scatter_size=50, future_end=2050
    )
