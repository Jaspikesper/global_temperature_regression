"""
infer.py
--------
Historical reconstruction plot with optional hover interactivity.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("TkAgg")               # force interactive backend
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"

from interactivity import attach_hover


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
    # --- Sanity checks --------------------------------------------------
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    years_all = np.arange(start_year, end_year + 1)
    preds = fitted_model(years_all)

    # --- Prepare canvas -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ffffff")
    ax.plot(years_all, preds, label="Model prediction", zorder=3)

    # Labels / legend ----------------------------------------------------
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title or "Historical inference",
    )
    ax.legend()

    # Hover interactivity ------------------------------------------------
    if interactive:
        attach_hover(
            ax,
            xobs,
            yobs,
            fitted_model,
            scatter_size=scatter_size,
            start=start_year,
            end=end_year,
        )

    plt.tight_layout()
    return fig, ax
