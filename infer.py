"""
Module containing long_term_inference – now able to add the sun background.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                # keep everything interactive
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"
import matplotlib.image as mpimg


def long_term_inference(
        fitted_model: callable,
        xobs: np.ndarray,
        yobs: np.ndarray,
        *,
        start_year: int = 1000,
        end_year: int = 2025,
        title: str | None = None,
        xlabel: str = "Year",
        ylabel: str = "Value",
        scatter_size: int = 50,
        show_background: bool = False,          # NEW
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot historical inference for a *fitted_model* over [start_year, end_year].

    Parameters
    ----------
    fitted_model : callable
        Function f(years) -> predicted values.
    xobs, yobs : np.ndarray
        Observed data used in the original fit; plotted here for reference.
    start_year, end_year : int
        Inclusive time window.
    show_background : bool
        If True, paint the sun image behind the graph.
    """
    # --- Sanity checks -----------------------------------------------------
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    years_all = np.arange(start_year, end_year + 1)
    preds = fitted_model(years_all)

    # --- Prepare canvas ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ffffff")
    ax.plot(years_all, preds, label="Model prediction", zorder=3)


    # Scatter observed points that fall in range
    in_range = (xobs >= start_year) & (xobs <= end_year)

    # Insert background LAST so it stays behind everything
    if show_background:

        try:
            bg = mpimg.imread("static/example2.png")
            x0 = np.min(xobs)
            x1 = np.max()
            y0 = np.min(yobs)
            y1 = np.max(yobs)
            ax.imshow(
                bg,
                aspect="auto",
                extent=(x0, 1.02*x1, y0, y1),
                zorder=0,
                a=1
            )
            print(x0, x1, y0, y1)
        except Exception as exc:
            print(f"[long_term_inference] Warning: could not load background: {exc}")

    # Labels / legend -------------------------------------------------------
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title or "Historical inference",
    )
    ax.legend()

    # ── Hover interactivity ------------------------------------------------
    halo, = ax.plot(
        [], [], "o",
        ms=np.sqrt(scatter_size) * 2,
        mfc="none", mec="yellow", mew=2, zorder=5,
    )
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        zorder=6,
    )
    ann.set_visible(False)

    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            ann.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        year = int(round(event.xdata))
        year = np.clip(year, start_year, end_year)
        pred_val = float(fitted_model(year))

        # If we actually have an observation for that year, use it;
        # otherwise just display the prediction.
        mask = xobs == year
        if mask.any():
            obs_val = float(yobs[mask][0])
            ann_text = f"Year : {year}\nObs  : {obs_val:.2f}\nPred: {pred_val:.2f}"
            halo_y = obs_val
        else:
            ann_text = f"Year : {year}\nPred: {pred_val:.2f}"
            halo_y = pred_val

        ann.xy = (year, halo_y)
        ann.set_text(ann_text)
        halo.set_data([year], [halo_y])
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    plt.tight_layout()
    return fig, ax
