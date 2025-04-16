import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess  # Import LOESS from statsmodels
from data_loader import x, y

plt.rcParams['backend'] = 'TkAgg'


def plot_interactive_regression(x, y, fit_func, background_img_path=None,
                                title="Interactive Regression Plot",
                                xlabel="Year",
                                ylabel="y",
                                scatter_label="Data",
                                regression_label="Regression Fit",
                                scatter_kwargs=None,
                                line_kwargs=None,
                                scatter_size=50,
                                future_end=2050):  # Extend future predictions up to a target year
    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}

    # Pre-compute model fit once using the provided fitting function
    model = fit_func(x, y)
    y_pred = model(x)

    last_x = np.max(x)
    # Prepare future prediction data if the last year is before future_end
    if last_x < future_end:
        future_x = np.linspace(last_x, future_end, 100)
        future_y = model(future_x)
    else:
        future_x, future_y = None, None

    # Create lookup table for hover interactions on historical data
    x_to_idx = {val: idx for idx, val in enumerate(x)}

    # Create figure and axes
    fig, ax = plt.subplots()

    # Background image setup - only load if provided
    if background_img_path:
        try:
            bg_img = mpimg.imread(background_img_path)

            y_data_min = np.min(y)
            y_data_max = np.max(y)

            ax.imshow(
                bg_img,
                aspect='auto',
                extent=[np.min(x), future_end, y_data_min, y_data_max],
                zorder=0
            )

            # Update the axes limits to include the full future range
            ax.set_xlim(np.min(x), future_end)
            ax.set_ylim(y_data_min, y_data_max)

        except Exception as e:
            print(f"Warning: Could not load background image from {background_img_path}: {e}")

    # Plot historical data and LOESS fit
    scatter = ax.scatter(x, y, s=scatter_size, c='black', label=scatter_label, zorder=2, **scatter_kwargs)
    regression_line, = ax.plot(x, y_pred, color='orange', linewidth=2, label=regression_label, zorder=3, **line_kwargs)

    # Plot future predictions if available, with a dashed style
    if future_x is not None and future_y is not None:
        future_line, = ax.plot(future_x, future_y, color='orange', linestyle='dashed',
                               linewidth=2, label="Future Prediction", zorder=3, **line_kwargs)
        # Add vertical dotted line at the boundary with the same orange as the regression line
        ax.axvline(last_x, color='orange', linestyle='dotted', linewidth=1)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)

    # Create a hover annotation for interactive display
    hover_annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    hover_annotation.set_visible(False)

    # Halo marker to highlight the hovered point
    halo, = ax.plot([], [], 'o', ms=np.sqrt(scatter_size), mec='yellow',
                    mfc='none', mew=2, zorder=4)

    current_x = None
    current_y = None

    # Last historical x-value for reference in the hover function
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

        # For historical data: show annotation for actual data points using the nearest integer x-value
        if x_val <= last_hist_x:
            hovered_x = int(round(x_val))
            if hovered_x in x_to_idx:
                idx = x_to_idx[hovered_x]
                true_x, true_y = x[idx], y[idx]
                pred_y = y_pred[idx]
                if true_x != current_x or true_y != current_y:
                    current_x, current_y = true_x, true_y
                    halo.set_data([true_x], [true_y])
                    hover_annotation.xy = (true_x, pred_y)
                    hover_annotation.set_text(f"Year = {true_x}\ny = {pred_y:.2f}")
                    hover_annotation.set_visible(True)
                    fig.canvas.draw_idle()
            elif hover_annotation.get_visible():
                current_x, current_y = None, None
                hover_annotation.set_visible(False)
                halo.set_data([], [])
                fig.canvas.draw_idle()
        else:
            # For the future region: compute the predicted value directly
            pred_y = model(x_val)
            if current_x != x_val or current_y != pred_y:
                current_x, current_y = x_val, pred_y
                halo.set_data([x_val], [pred_y])
                hover_annotation.xy = (x_val, pred_y)
                hover_annotation.set_text(f"Year = {x_val:.2f}\ny = {pred_y:.2f}")
                hover_annotation.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()

    return fig, ax, model


def general_curve_fit(model_func, p0=None, maxfev=2000):
    """
    A general curve fitting function using scipy.optimize.curve_fit.
    (Note: LOESS is non-parametric, so we won't use this for LOESS but keep it for other models.)
    """

    def fit(x, y):
        params, _ = curve_fit(model_func, x, y, p0=p0, maxfev=maxfev)
        return lambda x_new: model_func(x_new, *params)

    return fit


def loess_fit(x, y, frac=0.3):
    """
    Fit a LOESS curve to the data using statsmodels' lowess function.

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        frac (float): The fraction of the data used when estimating each y-value.

    Returns:
        A function that interpolates the LOESS-smoothed values so that it can be called with new x-values.
    """
    # Calculate the LOESS smooth; sort the output by x
    loess_result = lowess(y, x, frac=frac, return_sorted=True)
    # Create an interpolation function from the LOESS result
    return lambda x_new: np.interp(x_new, loess_result[:, 0], loess_result[:, 1])


# Example usage with a LOESS smoother
if __name__ == '__main__':
    np.random.seed(0)

    # Instead of using a parametric model (e.g., polynomial or exponential),
    # we create a LOESS fit by wrapping the loess_fit function with a fixed fraction.
    fit_func = lambda x, y: loess_fit(x, y, frac=0.3)

    fig, ax, model = plot_interactive_regression(
        x, y, fit_func,
        background_img_path='example2.png',
        title="Interactive LOESS Regression Plot",
        xlabel="Year",
        ylabel="Temperature",
        scatter_label="Temperature",
        regression_label="LOESS Fit",
        scatter_size=50,
        future_end=2050  # Extend future predictions up to the year 2050
    )
