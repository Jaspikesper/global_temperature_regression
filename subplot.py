#!/usr/bin/env python3
"""
Interactive Regression Explorer

Dependencies:
    • numpy
    • scipy
    • statsmodels
    • matplotlib (TkAgg backend)
    • tkinter (built‑in)
    • data_loader module with:
        - load_temperature_data() -> (x: np.ndarray, y: np.ndarray)
        - load_co2_data()       -> (x: np.ndarray, y: np.ndarray)
        - load_gis_data()       -> (x: np.ndarray, y: np.ndarray)
"""

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'

import tkinter as tk
from tkinter import ttk, messagebox
from functools import partial

# ── Fit functions ──────────────────────────────────────────────────────────────
def poly_fit(degree: int):
    """Return a function that fits a polynomial of given degree."""
    def fit(x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit

def exp_fit(x: np.ndarray, y: np.ndarray):
    """Fit an exponential model y = c·exp(r·x) + b."""
    def model(x_vals: np.ndarray, c: float, r: float, b: float):
        return c * np.exp(r * x_vals) + b
    params, _ = curve_fit(model, x, y, p0=(y[0], 0.01, 0))
    return lambda x_new: model(x_new, *params)

def loess_fit(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    """Fit a LOESS smoother and interpolate."""
    sm = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = sm[:,0], sm[:,1]
    return lambda x_new: np.interp(x_new, xs, ys)

# ── Plotting helper ────────────────────────────────────────────────────────────
def plot_regression(x: np.ndarray,
                    y: np.ndarray,
                    fit_func,
                    title: str,
                    xlabel: str,
                    ylabel: str,
                    scatter_size: int = 50,
                    future_end: int = None):
    """Scatter and regression line with interactivity."""
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=scatter_size, label='Data', zorder=2)
    model = fit_func(x, y)
    last_x = int(x.max())
    x_line = np.arange(x.min(), (future_end or last_x) + 1)
    y_line = model(x_line)
    ax.plot(x_line, y_line, label='Fit', zorder=3)
    if future_end and future_end > last_x:
        ax.plot(np.arange(last_x + 1, future_end + 1), model(np.arange(last_x + 1, future_end + 1)),
                linestyle='--', label='Future', zorder=3)
        ax.axvline(last_x, linestyle=':', linewidth=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()

    # Add annotation and halo
    ann = ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points',
                      bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle='->'))
    ann.set_visible(False)
    halo, = ax.plot([], [], 'o', ms=np.sqrt(scatter_size)*2, mec='yellow', mfc='none', mew=2, zorder=4)

    def on_move(event):
        if event.inaxes != ax:
            ann.set_visible(False)
            halo.set_data([], [])
            fig.canvas.draw_idle()
            return
        year = int(round(event.xdata))
        min_year = int(x.min())
        max_year = future_end if future_end else last_x
        year = max(min(year, max_year), min_year)
        idx = np.where(x == year)[0]
        if idx.size > 0:
            obs_str = f"{y[idx[0]]:.2f}"
            y_val = y[idx[0]]
        else:
            obs_str = '-'
            y_val = model(year)
        pred = model(year)
        ann.xy = (year, y_val)
        ann.set_text(f"Year: {year}\nObserved: {obs_str}\nPredicted: {pred:.2f}")
        halo.set_data([year], [y_val])
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()

# ── GUI application ───────────────────────────────────────────────────────────
class RegressionApp(tk.Tk):
    """Main window: select dataset, method, and options."""
    def __init__(self):
        super().__init__()
        self.title("Regression Explorer")

        # Map button labels → data loader functions
        import data_loader
        self.datasets = {
            "Temperature": data_loader.load_temperature_data,
            "CO2 & Temp":  data_loader.load_co2_data,
            "GIS":         data_loader.load_gis_data,
        }

        # Map method names → fitting callables
        self.methods = {
            "Linear":      poly_fit(1),
            "Quadratic":   poly_fit(2),
            "Cubic":       poly_fit(3),
            "Quartic":     poly_fit(4),
            "Exponential": exp_fit,
            "LOESS":       loess_fit,
        }

        self._build_ui()

    def _build_ui(self):
        # Dataset selection buttons
        ds_frame = ttk.LabelFrame(self, text="Dataset")
        ds_frame.pack(fill='x', padx=10, pady=5)
        for name, loader in self.datasets.items():
            btn = ttk.Button(ds_frame, text=name,
                             command=partial(self._on_plot, loader))
            btn.pack(side='left', padx=5)

        # Control panel
        ctrl = ttk.Frame(self)
        ctrl.pack(fill='x', padx=10, pady=5)

        ttk.Label(ctrl, text="Method:").grid(row=0, column=0, sticky='w')
        self.method_var = tk.StringVar(value="Linear")
        ttk.Combobox(ctrl, textvariable=self.method_var,
                     values=list(self.methods.keys()),
                     state='readonly') \
            .grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Future End:").grid(row=1, column=0, sticky='w')
        self.future_var = tk.IntVar(value=2050)
        ttk.Entry(ctrl, textvariable=self.future_var, width=10) \
            .grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="Scatter Size:").grid(row=2, column=0, sticky='w')
        self.scatter_var = tk.IntVar(value=50)
        ttk.Entry(ctrl, textvariable=self.scatter_var, width=10) \
            .grid(row=2, column=1, padx=5, pady=2)

        ttk.Button(self, text="Quit", command=self.destroy) \
            .pack(pady=10)

    def _on_plot(self, loader_func):
        try:
            x, y = loader_func()
        except Exception as e:
            return messagebox.showerror("Load Error", str(e))

        fit = self.methods[self.method_var.get()]
        plot_regression(
            x, y,
            fit,
            title=f"{self.method_var.get()} Fit",
            xlabel="Year",
            ylabel="Value",
            scatter_size=self.scatter_var.get(),
            future_end=self.future_var.get()
        )

if __name__ == "__main__":
    RegressionApp().mainloop()