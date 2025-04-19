import numpy as np
import data_loader
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"


def long_term_inference(
        fitted_model: callable,
        xobs: np.ndarray,
        yobs: np.ndarray,
        *,
        start_year: int = 1000,
        end_year: int = 1950,
        title: str | None = None,
        xlabel: str = "Year",
        ylabel: str = "Value",
        scatter_size: int = 60,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot historical inference for a *fitted_model* over [start_year, end_year].

    Parameters
    ----------
    fitted_model : callable
        Function f(years) -> predicted values, typically returned by one of
        your fit helpers (poly_fit, exp_fit, loess_fit, …).
    xobs, yobs : np.ndarray
        Original observations already used to fit the model.
        Used here only for plotting/hover feedback.
    start_year, end_year : int
        Inclusive range for the time axis (can extend beyond xobs).
    title : str | None
        Plot title; defaults to “Historical inference”.
    xlabel, ylabel : str
        Axis labels.
    scatter_size : int
        Size of observation markers.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (so you can tweak further).
    """
    # --- Sanity checks / coercions -----------------------------------------
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    years_all = np.arange(start_year, end_year + 1)
    preds     = fitted_model(years_all)

    # --- Prepare canvas -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ffffff")

    # Main curve & optional shading of extrapolated zone
    ax.plot(years_all, preds, label="Model prediction", zorder=2)

    from data_loader import _load



    # Indicate the observation envelope
    x_min_obs, x_max_obs = int(xobs.min()), int(xobs.max())
    if start_year < x_min_obs:
        ax.axvspan(start_year, x_min_obs, color="#b3cde3", alpha=.08,
                   label="extrapolated zone (pre‑obs)")
    if end_year > x_max_obs:
        ax.axvspan(x_max_obs, end_year, color="#fbb4ae", alpha=.08,
                   label="extrapolated zone (post‑obs)")

    # Scatter observed points (only those falling in the [start, end] range)
    in_range = (xobs >= start_year) & (xobs <= end_year)
    ax.scatter(xobs[in_range], yobs[in_range],
               s=scatter_size, c="black", label="Observed", zorder=3)

    # Labels / legend
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title or "Historical inference")
    ax.legend()

    # --- Hover / halo interactivity ----------------------------------------
    halo, = ax.plot([], [], "o", ms=np.sqrt(scatter_size) * 2,
                    mfc="none", mec="yellow", mew=2, zorder=4)
    ann = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                      textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"), zorder=5)
    ann.set_visible(False)

    # Pre‑compute dict for O(1) observed lookup
    obs_lookup = {int(x): float(y) for x, y in zip(xobs, yobs)}

    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            ann.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        year = int(round(event.xdata))
        year = max(min(year, end_year), start_year)

        # Observed?
        if year in obs_lookup:
            obs_val = obs_lookup[year]
            disp_obs = f"{obs_val:.2f}"
            y_val = obs_val
        else:
            disp_obs = "–"
            y_val = fitted_model(year)

        ann.xy = (year, y_val)
        ann.set_text(f"Year : {year}\nObs  : {disp_obs}\nPred: {fitted_model(year):.2f}")
        halo.set_data([year], [y_val])
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    plt.tight_layout()
    return fig, ax
