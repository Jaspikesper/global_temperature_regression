#!/usr/bin/env python3
"""
Interactive Regression Explorer – sun‑fixed edition
(extended to run a post‑GUI historical inference)
"""

import numpy as np
from functools import partial
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
import data_loader

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import ttk, messagebox

from infer import long_term_inference

# ── Globals to store selected dataset info ──
dataset_name: str = ""
xmax: float = 0.0
ymax: float = 0.0

DATASET_KEYS = {
    "Temperature": "temperature",
    "CO2 & Temp":  "co2",
    "GIS":         "gis",
}

def poly_fit(degree: int):
    def fit(x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit

def exp_fit(x: np.ndarray, y: np.ndarray):
    """
    Fit y ≈ c * exp(r*(x - x0)) + b, with data-driven initial guess.
    """
    x0 = x[0]

    def _model(x_vals, c, r, b):
        return c * np.exp(r * (x_vals - x0)) + b

    # 1. Baseline shift
    b0 = float(np.min(y))
    # 2. Starting amplitude
    c0 = float(y[0] - b0)
    # 3. Rate estimate (guard against non-positive)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)

    try:
        params, _ = curve_fit(
            _model,
            x, y,
            p0=p0,
            maxfev=5000,
        )
        return lambda x_new: _model(x_new, *params)
    except Exception as exc:
        print(f"[exp_fit] Fit failed with p0={p0}: {exc}")
        # fallback to flat model
        mean_y = float(np.mean(y))
        return lambda x_new: np.full_like(x_new, mean_y, dtype=float)


def loess_fit(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = smoothed[:, 0], smoothed[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)

def plot_regression(
        x: np.ndarray,
        y: np.ndarray,
        fit_func,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        scatter_size: int = 50,
        future_end: int | None = None,
        show_background: bool = False,
        interactive: bool = True,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#ffffff")

    model = fit_func(x, y)

    last_x = int(x.max())
    x_line = np.arange(x.min(), (future_end or last_x) + 1)
    ax.plot(x_line, model(x_line), label="Fit", zorder=3)

    if future_end and future_end > last_x:
        fut_x = np.arange(last_x + 1, future_end + 1)
        ax.plot(fut_x, model(fut_x), "--", label="Future", zorder=3)
        ax.axvline(last_x, linestyle=":", linewidth=1)

    ax.scatter(x, y, s=scatter_size, label="Data", zorder=4)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()

    if show_background:
        try:
            bg_img = mpimg.imread("static/example2.png")
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.imshow(bg_img, aspect="auto", extent=(x0, x1, y0, y1), zorder=0, alpha=1)
        except Exception as exc:
            print(f"[plot_regression] Warning: could not load background: {exc}")

    if interactive:
        halo, = ax.plot([], [], "o", ms=np.sqrt(scatter_size) * 2, mfc="none", mec="yellow", mew=2, zorder=5)
        ann = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"), zorder=6)
        ann.set_visible(False)

        def _on_move(event):
            if event.inaxes is not ax or event.xdata is None:
                ann.set_visible(False)
                halo.set_data([], [])
                fig.canvas.draw_idle()
                return

            year = int(round(event.xdata))
            year = np.clip(year, int(x.min()), future_end or last_x)
            idx = np.where(x == year)[0]

            if idx.size:
                obs_val = y[idx[0]]
                disp_obs = f"{obs_val:.2f}"
                y_val = obs_val
            else:
                disp_obs = "–"
                y_val = model(year)

            ann.xy = (year, y_val)
            ann.set_text(f"Year : {year}\nObs  : {disp_obs}\nPred: {model(year):.2f}")
            halo.set_data([year], [y_val])
            ann.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)

    plt.show()

class RegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Regression Explorer")
        self.geometry("450x300")

        self.datasets = {
            "Temperature": data_loader.load_temperature_data,
            "CO2 & Temp": data_loader.load_co2_data,
            "GIS": data_loader.load_gis_data,
        }
        self.methods = {
            "Linear": poly_fit(1),
            "Quadratic": poly_fit(2),
            "Cubic": poly_fit(3),
            "Quartic": poly_fit(4),
            "Exponential": exp_fit,
            "LOESS": loess_fit,
        }

        self._last_model = None
        self._last_x = None
        self._last_y = None
        self._last_show_bg = False
        self._last_interactive = True

        self._selected_name = next(iter(self.datasets))
        self._selected_loader = self.datasets[self._selected_name]

        self._build_ui()

    def _build_ui(self):
        ds_frame = ttk.LabelFrame(self, text="Dataset")
        ds_frame.pack(fill="x", padx=10, pady=5)
        for name in self.datasets:
            ttk.Button(ds_frame, text=name, command=partial(self._on_dataset_select, name)).pack(side="left", padx=5)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=10, pady=5)

        ttk.Label(ctrl, text="Method:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="Linear")
        ttk.Combobox(ctrl, textvariable=self.method_var, values=list(self.methods), state="readonly").grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Future End:").grid(row=1, column=0, sticky="w")
        self.future_var = tk.IntVar(value=2050)
        ttk.Entry(ctrl, textvariable=self.future_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Show Sun Background:").grid(row=2, column=0, sticky="w")
        self.bg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.bg_var).grid(row=2, column=1, sticky="w")

        ttk.Label(ctrl, text="Enable Interactivity:").grid(row=3, column=0, sticky="w")
        self.interactive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.interactive_var).grid(row=3, column=1, sticky="w")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Done", command=self._on_done).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.destroy).pack(side="left", padx=5)

    def _on_dataset_select(self, name):
        self._selected_name = name
        self._selected_loader = self.datasets[name]

    def _on_done(self):
        self._on_plot()
        self.after(100, self.destroy)

    def _on_plot(self):
        loader = self._selected_loader
        name = self._selected_name
        try:
            x, y = loader()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        global dataset_name, xmax, ymax
        dataset_name = DATASET_KEYS[name]
        xmax = float(np.max(x))
        ymax = float(np.max(y))

        fit_func = self.methods[self.method_var.get()]
        model = fit_func(x, y)
        self._last_model = model
        self._last_x = x
        self._last_y = y
        self._last_show_bg = bool(self.bg_var.get())
        self._last_interactive = bool(self.interactive_var.get())

        plot_regression(
            x, y,
            fit_func,
            title=f"{self.method_var.get()} Fit",
            xlabel="Year",
            ylabel="Value",
            scatter_size=50,
            future_end=self.future_var.get(),
            show_background=self._last_show_bg,
            interactive=self._last_interactive,
        )

    def get_last_fit(self):
        return self._last_model, self._last_x, self._last_y, self._last_show_bg, self._last_interactive
def run():
    app = RegressionApp()
    app.mainloop()

    model, x_obs, y_obs, show_bg, interactive = app.get_last_fit()  # Include interactive here
    if model is None:
        print("No plot was generated – nothing to infer. Exiting.")
        return

    print("Running historical inference …")
    x_long, y_long = data_loader.load_long_data()
    fig, ax = long_term_inference(
        model,
        x_long,
        y_long,
        start_year=1000,
        end_year=1950,
        title="Historical reconstruction (1000‑1950)",
        interactive=interactive,  # Crucial fix here
    )

    if show_bg:
        try:
            bg = mpimg.imread("static/example2.png")
            x0, _ = ax.get_xlim()
            y0_existing, _ = ax.get_ylim()
            y0 = min(y0_existing, float(y_long.min()), float(y_obs.min()))
            ax.set_ylim(y0, ymax)
            ax.imshow(bg, aspect="auto", extent=(x0, xmax, y0, ymax), zorder=0, alpha=1)
        except Exception as exc:
            print(f"[run] Warning: could not load background: {exc}")

    ax.scatter(x_long, y_long, s=25, c="blue", label="Long‑Term Temperature (Obs.)", zorder=4)
    ax.scatter(x_obs, y_obs, s=25, c="blue", alpha=0.5, zorder=4)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run()
