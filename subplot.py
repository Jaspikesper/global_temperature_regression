import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams['backend'] = 'TkAgg'

# ---------- Fit helper (LOESS) ----------
def loess_fit(x, y, frac=0.3):
    loess_res = lowess(y, x, frac=frac, return_sorted=True)
    return lambda x_new: np.interp(x_new, loess_res[:, 0], loess_res[:, 1])

# ---------- Grid of interactive plots ----------
def interactive_regression_grid(datasets, nrows=2, ncols=2,
                                scatter_size=50, future_end=2050):
    """
    datasets: list of tuples (x, y, fit_func, title, xlabel, ylabel)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    annotations = []
    halos = []
    plot_data = []

    # Plot each dataset on its own Axes
    for ax, (x, y, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x, y)
        y_pred = model(x)
        last_x = int(x.max())

        # future prediction
        if last_x < future_end:
            fx = np.linspace(last_x, future_end, 100)
            fy = model(fx)
        else:
            fx = fy = None

        ax.scatter(x, y, s=scatter_size, label="Data", zorder=2)
        ax.plot(x, y_pred, label="Fit", zorder=3)
        if fx is not None:
            ax.plot(fx, fy, linestyle='--', label="Future", zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)

        # prepare hover annotation + halo
        ann = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        ann.set_visible(False)
        annotations.append(ann)

        halo, = ax.plot([], [], 'o',
                        ms=np.sqrt(scatter_size),
                        mec='yellow', mfc='none', mew=2,
                        zorder=4)
        halos.append(halo)

        plot_data.append((x, y, model, last_x))

    # single event handler: halo snaps to nearest data point; annotation shows continuous model value
    def on_move(event):
        if event.xdata is None or event.inaxes is None:
            for ann, halo in zip(annotations, halos):
                ann.set_visible(False)
                halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        for ax, (x_arr, y_arr, model, last_x), ann, halo in zip(
                axes, plot_data, annotations, halos):
            if event.inaxes == ax:
                x_val = event.xdata
                # find index of nearest x data point
                idx = int(np.argmin(np.abs(x_arr - x_val)))
                snap_x = x_arr[idx]
                snap_y = y_arr[idx]

                # continuous prediction at true mouse x
                cont_y = model(x_val)

                # update halo at the exact data point
                halo.set_data([snap_x], [snap_y])
                # position annotation at that same data point
                ann.xy = (snap_x, snap_y)
                ann.set_text(f"Year = {snap_x}\nValue = {cont_y:.2f}")
                ann.set_visible(True)
                fig.canvas.draw_idle()
                return

        # if not over any subplot
        for ann, halo in zip(annotations, halos):
            ann.set_visible(False)
            halo.set_data([], [])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()
    return fig, axes

# ---------- Demo ----------
if __name__ == "__main__":
    np.random.seed(0)
    years = np.arange(2000, 2010)
    y1 = years * 0.1 + np.random.randn(len(years)) * 0.5
    y2 = np.sin((years - 2000) / 2) + np.random.randn(len(years)) * 0.1
    y3 = np.log1p(years - 1995) + np.random.randn(len(years)) * 0.1
    y4 = (years - 2000)**2 * 0.02 + np.random.randn(len(years)) * 0.5

    datasets = [
        (years, y1, loess_fit, "Linear‑ish Trend", "Year", "Value"),
        (years, y2, loess_fit, "Sinusoidal",       "Year", "Value"),
        (years, y3, loess_fit, "Log Growth",       "Year", "Value"),
        (years, y4, loess_fit, "Quadratic‑ish",    "Year", "Value"),
    ]

    interactive_regression_grid(datasets, nrows=2, ncols=2, future_end=2025)
