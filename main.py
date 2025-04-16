import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
                                scatter_size=50):
    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}

    # Pre-compute model fit once
    model = fit_func(x, y)
    y_pred = model(x)

    # Create lookup table for hover interactions
    x_to_idx = {val: idx for idx, val in enumerate(x)}

    # Create figure and axes only once
    fig, ax = plt.subplots()

    # Background image setup - only load if provided
    if background_img_path:
        try:
            # Load image more efficiently
            bg_img = mpimg.imread(background_img_path)
            ax.imshow(bg_img, aspect='auto', extent=[min(x), max(x), min(y), max(y)], zorder=0)
        except Exception:
            print(f"Warning: Could not load background image from {background_img_path}")

    # Plot data once
    scatter = ax.scatter(x, y, s=scatter_size, c='black', label=scatter_label, zorder=2, **scatter_kwargs)
    regression_line, = ax.plot(x, y_pred, color='orange', linewidth=2, label=regression_label, zorder=3, **line_kwargs)

    # Set plot aesthetics once
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    legend = ax.legend(fontsize=12)

    # Pre-create annotation objects
    hover_annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    hover_annotation.set_visible(False)

    halo, = ax.plot([], [], 'o', ms=np.sqrt(scatter_size), mec='yellow', mfc='none', mew=2, zorder=4)

    # Track current point for comparison
    current_x = None
    current_y = None

    # Optimize hover handler for speed
    def on_move(event):
        nonlocal current_x, current_y

        if event.inaxes != ax or event.xdata is None:
            if hover_annotation.get_visible():
                hover_annotation.set_visible(False)
                halo.set_data([], [])
                fig.canvas.draw_idle()
            return

        # Round to nearest integer for year lookup
        hovered_x = int(round(event.xdata))

        # Fast lookup using dict instead of where
        if hovered_x in x_to_idx:
            idx = x_to_idx[hovered_x]
            x_val, y_val = x[idx], y[idx]
            pred_y = y_pred[idx]

            # Update only if position changed - avoid empty array comparison
            if x_val != current_x or y_val != current_y:
                current_x, current_y = x_val, y_val
                halo.set_data([x_val], [y_val])  # Explicitly use lists to avoid empty array issues
                hover_annotation.xy = (x_val, pred_y)
                hover_annotation.set_text(f"Year = {hovered_x}\ny = {pred_y:.2f}")
                hover_annotation.set_visible(True)
                fig.canvas.draw_idle()
        elif hover_annotation.get_visible():
            current_x, current_y = None, None
            hover_annotation.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()

    # Connect the event handler
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()

    return fig, ax, model


# Example usage
if __name__ == '__main__':
    np.random.seed(0)


    # Define more efficient fit function to avoid repeated polyfit
    def quadratic_fit(x, y):
        coeffs = np.polyfit(x, y, deg=2)
        return lambda x_new: np.polyval(coeffs, x_new)


    fig, ax, model = plot_interactive_regression(
        x, y, quadratic_fit,
        background_img_path='example2.png',
        title="Interactive Quadratic Regression",
        xlabel="Year",
        ylabel="y",
        scatter_label="Temperature",
        regression_label="Quadratic Fit",
        scatter_size=50
    )