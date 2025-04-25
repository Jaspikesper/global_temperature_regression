"""
Module containing long_term_inference  simplified version without background.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                # force interactive backend
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"


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
        interactive: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot historical inference for a *fitted_model* over [start_year, end_year].

    Parameters
    ----------
    fitted_model : callable
        Function f(years) -> predicted values.
    xobs, yobs : np.ndarray
        Observations used in the original fit
    start_year, end_year : int
        Inclusive time window.
    interactive : bool
        If True, enable hover interactivity; otherwise, disable it.
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

    # Labels / legend -------------------------------------------------------
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title or "Historical inference",
    )
    ax.legend()

    #  Hover interactivity ------------------------------------------------
    if interactive:
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

            mask = xobs == year
            if mask.any():
                obs_val = float(yobs[mask][0])
                ann_text = f"Year : {year}\nObs  : {obs_val:.2f}\nPred: {pred_val:.2f}"
                halo_y = obs_val
            else:
                ann_text = f"Year : {year}\nPred: {pred_val:.2f}"
                halo_y = pred_val

            ann.xy = (year, halo_y)
            ann.set_text(ann_text)
            halo.set_data([year], [halo_y])
            ann.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)

    plt.tight_layout()
    return fig, ax
