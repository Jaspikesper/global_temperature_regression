"""
interactivity.py
----------------
Reusable yellow-halo hover helper for matplotlib Axes.

attach_hover(ax, x_obs, y_obs, predictor, *, scatter_size=50, start=None, end=None)
     ax           : target Axes
    x_obs, y_obs : 1-D NumPy arrays of observed data
     predictor    : callable(years) -> predictions
    scatter_size : radius basis for halo
    start / end  : clamp range for the cursor (defaults to data extents)

Returns (annotation, halo_line) so callers may further tweak style if desired.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def attach_hover(
    ax: plt.Axes,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    predictor,
    *,
    scatter_size: int = 50,
    start: int | None = None,
    end: int | None = None,
):
    fig = ax.figure
    halo, = ax.plot(
        [], [],
        "o",
        ms=np.sqrt(scatter_size) * 2,
        mfc="none",
        mec="yellow",
        mew=2,
        zorder=5,
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

    xmin = int(np.min(x_obs)) if start is None else int(start)
    xmax = int(np.max(x_obs)) if end   is None else int(end)

    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            ann.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        yr = int(round(event.xdata))
        yr = max(xmin, min(xmax, yr))
        pred = float(predictor(yr))

        mask = x_obs == yr
        if mask.any():
            obs = float(y_obs[mask][0])
            halo_y = obs
            text = f"Year : {yr}\nObs  : {obs:.2f}\nPred: {pred:.2f}"
        else:
            halo_y = pred
            text = f"Year : {yr}\nPred: {pred:.2f}"

        ann.xy = (yr, halo_y)
        ann.set_text(text)
        halo.set_data([yr], [halo_y])
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    return ann, halo
