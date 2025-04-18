#!/usr/bin/env python3
"""
Interactive Regression Explorer ‑‑ **sun‑fixed edition**

Dependencies
────────────────
    • numpy
    • scipy
    • statsmodels
    • matplotlib (TkAgg backend)
    • tkinter (built‑in)
    • data_loader module with:
        - load_temperature_data() -> (x, y)
        - load_co2_data()         -> (x, y)
        - load_gis_data()         -> (x, y)
    • static/useMe.png  ← a transparent PNG of the sun
"""

import numpy as np
from functools import partial
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use('TkAgg')                     # force interactive backend
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'

import matplotlib.image as mpimg
import tkinter as tk
from tkinter import ttk, messagebox


# ────────────── Fit helpers ──────────────

def poly_fit(degree: int):
    def fit(x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit

def exp_fit(x: np.ndarray, y: np.ndarray):
    def model(x_vals: np.ndarray, c, r, b):
        return c * np.exp(r * x_vals) + b
    params, _ = curve_fit(model, x, y, p0=(y[0], 0.01, 0))
    return lambda x_new: model(x_new, *params)

def loess_fit(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = smoothed[:, 0], smoothed[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)


# ────────────── Plot‑creation helper ──────────────

def plot_regression(x: np.ndarray,
                    y: np.ndarray,
                    fit_func,
                    *,
                    title: str,
                    xlabel: str,
                    ylabel: str,
                    scatter_size: int = 50,
                    future_end: int | None = None,
                    show_background: bool = False):

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

    ax.scatter(x, y, s=scatter_size, label="Data", zorder=2)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()

    # Force redraw to update limits before getting them
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()


    halo, = ax.plot([], [], "o", ms=np.sqrt(scatter_size) * 2,
                    mfc="none", mec="yellow", mew=2, zorder=4)
    ann = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                      textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"),
                      zorder=5)
    ann.set_visible(False)

    def on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            ann.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()
            return

        year = int(round(event.xdata))
        year = max(min(year, future_end or last_x), int(x.min()))
        idx = np.where(x == year)[0]

        if idx.size:
            obs_val = y[idx[0]]
            disp_obs = f"{obs_val:.2f}"
            y_val = obs_val
        else:
            disp_obs = "–"
            y_val = model(year)

        ann.xy = (year, y_val)
        ann.set_text(f"Year : {year}\nObs  : {disp_obs}\nPred: {model(year):.2f}")
        halo.set_data([year], [y_val])
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


class RegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Regression Explorer")

        import data_loader
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

        self._build_ui()

    def _build_ui(self):
        ds_frame = ttk.LabelFrame(self, text="Dataset")
        ds_frame.pack(fill="x", padx=10, pady=5)
        for name, loader in self.datasets.items():
            ttk.Button(ds_frame, text=name,
                       command=partial(self._on_plot, loader)).pack(side="left", padx=5)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=10, pady=5)

        ttk.Label(ctrl, text="Method:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="Linear")
        ttk.Combobox(ctrl, textvariable=self.method_var,
                     values=list(self.methods),
                     state="readonly").grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Future End:").grid(row=1, column=0, sticky="w")
        self.future_var = tk.IntVar(value=2050)
        ttk.Entry(ctrl, textvariable=self.future_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Scatter Size:").grid(row=2, column=0, sticky="w")
        self.scatter_var = tk.IntVar(value=50)
        ttk.Entry(ctrl, textvariable=self.scatter_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Show Sun Background:").grid(row=3, column=0, sticky="w")
        self.bg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, variable=self.bg_var).grid(row=3, column=1, sticky="w")

        ttk.Button(self, text="Quit", command=self.destroy).pack(pady=10)

    def _on_plot(self, loader):
        try:
            x, y = loader()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        fit = self.methods[self.method_var.get()]
        plot_regression(
            x, y, fit,
            title=f"{self.method_var.get()} Fit",
            xlabel="Year", ylabel="Value",
            scatter_size=self.scatter_var.get(),
            future_end=self.future_var.get(),
            show_background=self.bg_var.get(),
        )


if __name__ == "__main__":
    RegressionApp().mainloop()