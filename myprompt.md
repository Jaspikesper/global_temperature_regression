data_loader.py
---
# data_loader.py
"""CSV loader helpers returns (x, y) NumPy arrays for the three datasets."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).with_suffix('').parent / 'data'   # ./data alongside this file

# dataset name(csv filename, xcolumn, ycolumn)
_FILES = {
    'temperature': ('Temperature_Data.csv', 'year', 'temperature_anomaly'),
    'co2':         ('merged_co2_temp.csv', 'year', 'temperature_anomaly'),  # rename cols if different
    'gis':         ('gistemp.csv',        'year', 'temperature_anomaly'),
    'long':        ('long.csv',          'year', 'temperature_anomaly')
}

def _load(key):
    """Internal: read CSV, lowercase headers, return requested columns as NumPy arrays."""
    csv, xcol, ycol = _FILES[key]
    df = pd.read_csv(DATA_DIR / csv).rename(str.lower, axis=1)
    return df[xcol].to_numpy(), df[ycol].to_numpy()

# public loaders
load_temperature_data = lambda: _load('temperature')
load_co2_data         = lambda: _load('co2')
load_gis_data         = lambda: _load('gis')
load_long_data        = lambda: _load('long')




---
infer.py
---
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


---
interactivity.py
---
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


---
main.py
---
#!/usr/bin/env python3
"""
main.py
-------
Two-by-two interactive grid comparing regression fits.
"""

from __future__ import annotations

import sys
import numpy as np

# --- matplotlib bootstrap ----------------------------------------------
def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.stderr.write(
            "Error: matplotlib is required for plotting but is not installed.\n"
            "Please install it via 'pip install matplotlib'.\n"
        )
        sys.exit(1)


plt = _import_matplotlib()
plt.rcParams["backend"] = "TkAgg"

from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from interactivity import attach_hover
from data_loader import load_temperature_data, load_long_data


# ---------- Fit helpers -------------------------------------------------
def poly_fit(degree):
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exp_fit(x: np.ndarray, y: np.ndarray):
    x0 = x[0]

    def _model(x_vals, c, r, b):
        return c * np.exp(r * (x_vals - x0)) + b

    b0 = float(np.min(y))
    c0 = float(y[0] - b0)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=5000)
        return lambda x_new: _model(x_new, *params)
    except Exception as exc:
        print(f"[exp_fit] Fit failed with p0={p0}: {exc}")
        mean_y = float(np.mean(y))
        return lambda x_new: np.full_like(x_new, mean_y, dtype=float)


def loess_fit(x_arr, y_arr, frac=0.3):
    loess_res = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    xs, ys = loess_res[:, 0], loess_res[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)


# ---------- Interactive grid helper ------------------------------------
def interactive_regression_grid(
    datasets,
    nrows=2,
    ncols=2,
    scatter_size=50,
    future_end=2050,
    recent_data=None,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label="Data", zorder=2)
        if recent_data and 'Extrapolation' in title:
            x_recent, y_recent = recent_data
            ax.scatter(x_recent, y_recent, s=scatter_size, label='Recent Data', zorder=2)

        ax.plot(x_arr, y_pred, label='Fit', zorder=3)
        if last_x < future_end:
            fx = np.arange(last_x, int(future_end) + 1)
            ax.plot(fx, model(fx), linestyle='--', label='Future', zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # Re-use consolidated hover logic
        attach_hover(
            ax,
            x_arr,
            y_arr,
            model,
            scatter_size=scatter_size,
            start=int(x_arr.min()),
            end=int(future_end),
        )

    plt.tight_layout()
    plt.show()
    return fig, axes


# -------------------------- CLI entry -----------------------------------
if __name__ == '__main__':
    x_temp, y_temp = load_temperature_data()
    x_long, y_long = load_long_data()

    quadratic_fit_func = poly_fit(2)
    quad_model = quadratic_fit_func(x_temp, y_temp)

    exp_fit_func = exp_fit
    exp_model = exp_fit_func(x_temp, y_temp)

    datasets = [
        (x_temp, y_temp, quadratic_fit_func, 'Quadratic Fit (good recently / bad long-term fit)', '', 'Temperature'),
        (x_temp, y_temp, exp_fit_func, 'Exponential Fit (good recently / good long-term fit)', '', 'Temperature'),
        (x_long, y_long, lambda *_: quad_model, 'Extrapolation (Quadratic)', 'Year', 'Temperature'),
        (x_long, y_long, lambda *_: exp_model, 'Extrapolation (Exponential)', 'Year', 'Temperature'),
    ]

    interactive_regression_grid(
        datasets,
        nrows=2,
        ncols=2,
        scatter_size=30,
        future_end=2050,
        recent_data=(x_temp, y_temp),
    )


---
myprompt.md
---
data_loader.py
---
# data_loader.py
"""CSV loader helpers returns (x, y) NumPy arrays for the three datasets."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).with_suffix('').parent / 'data'   # ./data alongside this file

# dataset name(csv filename, xcolumn, ycolumn)
_FILES = {
    'temperature': ('Temperature_Data.csv', 'year', 'temperature_anomaly'),
    'co2':         ('merged_co2_temp.csv', 'year', 'temperature_anomaly'),  # rename cols if different
    'gis':         ('gistemp.csv',        'year', 'temperature_anomaly'),
    'long':        ('long.csv',          'year', 'temperature_anomaly')
}

def _load(key):
    """Internal: read CSV, lowercase headers, return requested columns as NumPy arrays."""
    csv, xcol, ycol = _FILES[key]
    df = pd.read_csv(DATA_DIR / csv).rename(str.lower, axis=1)
    return df[xcol].to_numpy(), df[ycol].to_numpy()

# public loaders
load_temperature_data = lambda: _load('temperature')
load_co2_data         = lambda: _load('co2')
load_gis_data         = lambda: _load('gis')
load_long_data        = lambda: _load('long')




---
infer.py
---
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


---
interactivity.py
---
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


---
main.py
---
#!/usr/bin/env python3
"""
main.py
-------
Two-by-two interactive grid comparing regression fits.
"""

from __future__ import annotations

import sys
import numpy as np

# --- matplotlib bootstrap ----------------------------------------------
def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.stderr.write(
            "Error: matplotlib is required for plotting but is not installed.\n"
            "Please install it via 'pip install matplotlib'.\n"
        )
        sys.exit(1)


plt = _import_matplotlib()
plt.rcParams["backend"] = "TkAgg"

from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from interactivity import attach_hover
from data_loader import load_temperature_data, load_long_data


# ---------- Fit helpers -------------------------------------------------
def poly_fit(degree):
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exp_fit(x: np.ndarray, y: np.ndarray):
    x0 = x[0]

    def _model(x_vals, c, r, b):
        return c * np.exp(r * (x_vals - x0)) + b

    b0 = float(np.min(y))
    c0 = float(y[0] - b0)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=5000)
        return lambda x_new: _model(x_new, *params)
    except Exception as exc:
        print(f"[exp_fit] Fit failed with p0={p0}: {exc}")
        mean_y = float(np.mean(y))
        return lambda x_new: np.full_like(x_new, mean_y, dtype=float)


def loess_fit(x_arr, y_arr, frac=0.3):
    loess_res = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    xs, ys = loess_res[:, 0], loess_res[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)


# ---------- Interactive grid helper ------------------------------------
def interactive_regression_grid(
    datasets,
    nrows=2,
    ncols=2,
    scatter_size=50,
    future_end=2050,
    recent_data=None,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (x_arr, y_arr, fit_func, title, xlabel, ylabel) in zip(axes, datasets):
        model = fit_func(x_arr, y_arr)
        y_pred = model(x_arr)
        last_x = int(np.max(x_arr))

        ax.scatter(x_arr, y_arr, s=scatter_size, label="Data", zorder=2)
        if recent_data and 'Extrapolation' in title:
            x_recent, y_recent = recent_data
            ax.scatter(x_recent, y_recent, s=scatter_size, label='Recent Data', zorder=2)

        ax.plot(x_arr, y_pred, label='Fit', zorder=3)
        if last_x < future_end:
            fx = np.arange(last_x, int(future_end) + 1)
            ax.plot(fx, model(fx), linestyle='--', label='Future', zorder=3)
            ax.axvline(last_x, linestyle=':', linewidth=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # Re-use consolidated hover logic
        attach_hover(
            ax,
            x_arr,
            y_arr,
            model,
            scatter_size=scatter_size,
            start=int(x_arr.min()),
            end=int(future_end),
        )

    plt.tight_layout()
    plt.show()
    return fig, axes


# -------------------------- CLI entry -----------------------------------
if __name__ == '__main__':
    x_temp, y_temp = load_temperature_data()
    x_long, y_long = load_long_data()

    quadratic_fit_func = poly_fit(2)
    quad_model = quadratic_fit_func(x_temp, y_temp)

    exp_fit_func = exp_fit
    exp_model = exp_fit_func(x_temp, y_temp)

    datasets = [
        (x_temp, y_temp, quadratic_fit_func, 'Quadratic Fit (good recently / bad long-term fit)', '', 'Temperature'),
        (x_temp, y_temp, exp_fit_func, 'Exponential Fit (good recently / good long-term fit)', '', 'Temperature'),
        (x_long, y_long, lambda *_: quad_model, 'Extrapolation (Quadratic)', 'Year', 'Temperature'),
        (x_long, y_long, lambda *_: exp_model, 'Extrapolation (Exponential)', 'Year', 'Temperature'),
    ]

    interactive_regression_grid(
        datasets,
        nrows=2,
        ncols=2,
        scatter_size=30,
        future_end=2050,
        recent_data=(x_temp, y_temp),
    )


---
project_high-levels\math_club_notes.md
---
# Earth Day To-Do List

## Earth Day Presentation
- **Date**: April 22nd @ 12:30â€“2:30 PM
- **Everything Due**: 4/20/25

### Tasks
- Practice presentations
  - Roman will work on presentation & share with Jasper
- Decide on who presents parts of the presentation
- Chris will judge the presentation and use ChatGPT to generate tough questions
- Create tri-fold poster
  - Visualize MSE equation

---

## Program
- **Merge programs into 1 program**
  - Be able to sweep data and graph
- **Merge data into 1 folder**
- **Mess level interactivity** (if time allows)
- **Create a GUI**
  - Allow for Alt+Tab
  - If GUI fails, then rely on Alt+Tab

---

## Roles
- **Roman's Roles**: Slideshow, Data Organization  
- **Chris' Role**: Judge presentation  
- **Jasper's Roles**: Data Analytics, Poster


---
project_high-levels\monroes_motivated_sequence.md
---
## Monroe's Motivated Sequence: Climate Change Argument

### 1. Attention
**Climate change is happening. Fighting climate change is urgent.**

### 2. Need
**While temperatures have risen and fallen slightly in the past, more recent data shows an accelerating increase. These hotter temperatures cause disasters, including wildfires and coastal flooding. **

### 3. Satisfaction
**Examine and seek to explain data through empirical risk minimization study. The task is not simply find the closest curve but the one that: 1: accurately explains past data and 2: makes accurate predictions.**

### 4. Visualization
**Display interactive regression plotting project, allow audience members to express their own predictions.**

### 5. Action
**Understand that climate change is a long term issue, that is exponential in nature. Don't be afraid to call out climate denial. Fight falsehoods with facts, not hatred or arrogance. Back the science and secure funding for climate research even when politics makes it harder. Use your role in society to fight emissions. **


---
project_high-levels\rubric.md
---
# Environmental Display Rating Rubric

| Category                                                                        | Possible POINTS | TOTAL   | Breakdown of the Criteria and Rating System                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------- | --------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Environmental Impact Statement & "Our Power, Our Planet" Theme Incorporation | 25              | \_\_/25 | â€¢ <strong>Accuracy:</strong> The display of relevant information concerning environmental issues and sustainability.<br><br>â€¢ <strong>Theme Relation:</strong> Relevancy in the reflection of the theme "Our Power, Our Planet".                                                                                 |
| 2. Readability and Organization                                                 | 25              | \_\_/25 | â€¢ <strong>Clarity & Structure:</strong> Information is presented with clear, readable headings, labels, and text, arranged in a format that guides the viewer through the content seamlessly.                                                                                                                  |
| 3. Visual Aesthetics and Composition                                            | 25              | \_\_/25 | â€¢ <strong>Aesthetic Quality:</strong> The presentation of appealing design and colors that engage viewers effectively.<br><br>â€¢ <strong>Composition:</strong> The design is well-organized and inviting.                                                                                                       |
| 4. Creativity & Audience Engagement                                             | 25              | \_\_/25 | â€¢ <strong>Impact:</strong> It effectively conveys information, promoting actionable insights and encouraging a call to action.<br><br>â€¢ <strong>Originality:</strong> The display features unique and original works.<br><br>â€¢ <strong>Creativity:</strong> Utilising props or interactive components to enhance the display and showcase resourcefulness. |

# Earth Day Winner: \$300 Club Fund Prize

## Criteria

The display must be the original work of the students of the club, demonstrate a clear message or raise awareness about the state of our environment, and reflect creative expression of the Earth Day theme.

---

**My vote goes to The Pride Club.**  
The students at the table were very welcoming. They encouraged me to make a little potted seedling and explained the purpose of their club. They talked about the importance of keeping our planet clean and GREEN. Their members were engaged in the activity, and it was nice watching them encourage other students to participate.

They also had a gender sticker basket of cute stickers to share with others. The advisor for the club was also close byâ€”encouraging, but letting the students lead the show.

**Five STARS!!!**


---
readme.md
---
ÿþ#   I n t e r a c t i v e   R e g r e s s i o n   E x p l o r e r   -   C l i m a t e   V i s u a l i z a t i o n   S u i t e 
 
 
 
 W e l c o m e   t o   t h e   M a t h   C l u b ' s   C l i m a t e   C h a n g e   P r o j e c t .   T h i s   t o o l k i t   l e t s   y o u   v i s u a l i z e   c l i m a t e   t r e n d s   i n t e r a c t i v e l y ,   o f f e r i n g   b o t h   a   G U I   a p p l i c a t i o n   a n d   a n   a n i m a t e d   m u l t i - f i t   e x p l o r e r .   E x p e r i m e n t   w i t h   d i f f e r e n t   r e g r e s s i o n   m o d e l s ,   e x p l o r e   d a t a s e t s ,   a n d   e v e n   o v e r l a y   a   c u s t o m   b a c k g r o u n d   i m a g e   ( l i k e   t h e   s u n ) . 
 
 
 
 - - - 
 
 
 
 # #   L a u n c h i n g   t h e   G U I 
 
 
 
 R u n   t h e   f o l l o w i n g   t o   l a u n c h   t h e   i n t e r a c t i v e   r e g r e s s i o n   G U I : 
 
 
 
 ` ` ` b a s h 
 
 p y t h o n   s u b p l o t . p y 
 
 ` ` ` 
 
 
 
 T h i s   o p e n s   a   w i n d o w   w h e r e   y o u   c a n : 
 
 
 
 -   C h o o s e   b e t w e e n   d a t a s e t s :   T e m p e r a t u r e ,   C O 2   &   T e m p e r a t u r e ,   a n d   G I S   r e c o r d s 
 
 -   S e l e c t   f i t t i n g   m e t h o d s :   L i n e a r ,   P o l y n o m i a l   ( u p   t o   Q u a r t i c ) ,   E x p o n e n t i a l ,   o r   L O E S S   s m o o t h i n g 
 
 -   C o n t r o l   h o w   f a r   i n t o   t h e   f u t u r e   t h e   r e g r e s s i o n   e x t r a p o l a t e s 
 
 -   A d j u s t   s c a t t e r - p o i n t   s i z e 
 
 -   T o g g l e   a   b a c k g r o u n d   i m a g e ,   i d e a l l y   s h a p e d   l i k e   a   s u n 
 
 
 
 M o u s e   o v e r   a n y   p o i n t   t o   s e e   r e a l - t i m e   t o o l t i p s   s h o w i n g   o b s e r v e d   v s .   p r e d i c t e d   v a l u e s .   D r a g   v e r t i c a l l y   t o   s e e   e f f e c t s   l i v e   i f   d r a g g i n g   i s   s u p p o r t e d   i n   y o u r   b u i l d . 
 
 
 
 N o t e :   I f   t h e   G U I   r u n s   s l o w l y   o r   b e c o m e s   u n r e s p o n s i v e ,   t e l l   J a s p e r   t o   o p t i m i z e   t h e   r e d r a w   l o o p   o r   l o w e r   t h e   s c a t t e r - p o i n t   c o u n t .   H e ' s   p r o b a b l y   a l r e a d y   g o t   a   f i x . 
 
 
 
 - - - 
 
 
 
 # #   R u n n i n g   t h e   G r i d   E x p l o r e r 
 
 
 
 T r y   t h e   i n t e r a c t i v e   r e g r e s s i o n   g r i d   i n   m a i n . p y ,   w h i c h   c o m p a r e s   m u l t i p l e   m o d e l s   s i d e - b y - s i d e : 
 
 
 
 ` ` ` b a s h 
 
 p y t h o n   m a i n . p y 
 
 ` ` ` 
 
 
 
 T h i s   t o o l : 
 
 
 
 -   D i s p l a y s   m u l t i p l e   r e g r e s s i o n   f i t s   ( L i n e a r ,   E x p o n e n t i a l ,   L O E S S ,   e t c . ) 
 
 -   R e s p o n d s   t o   m o u s e   h o v e r   w i t h   p r e c i s e   m o d e l   d i a g n o s t i c s 
 
 -   S u p p o r t s   r e a l - t i m e   p r e d i c t i o n   p a s t   o b s e r v e d   y e a r s   ( d a s h e d   l i n e ) 
 
 -   M a k e s   i t   e a s i e r   t o   c o m p a r e   m o d e l   q u a l i t y   a t   a   g l a n c e 
 
 
 
 - - - 
 
 
 
 # #   I m a g e   P r e p a r a t i o n 
 
 
 
 T o   i n c l u d e   a   c u s t o m   b a c k g r o u n d   ( l i k e   t h e   s u n ) ,   f o r m a t   y o u r   i m a g e   w i t h   a s p e c t _ r e s h a p e r . p y . 
 
 
 
 # # #   S t e p - b y - S t e p 
 
 
 
 1 .   P l a c e   y o u r   o r i g i n a l   i m a g e   i n   t h e   s t a t i c /   d i r e c t o r y .     
 
 2 .   E d i t   t h e   s c r i p t ' s   t o p   l i n e s   t o   r e f e r e n c e   y o u r   f i l e n a m e : 
 
 
 
       ` ` ` p y t h o n 
 
       i n p u t _ f i l e   =   ' s t a t i c / y o u r _ i m a g e . p n g ' 
 
       o u t p u t _ f i l e   =   ' s t a t i c / u s e M e . p n g ' 
 
       t a r g e t _ a s p e c t _ r a t i o   =   9   /   1 6 
 
       ` ` ` 
 
 
 
 3 .   R u n   t h e   s c r i p t : 
 
 
 
       ` ` ` b a s h 
 
       p y t h o n   a s p e c t _ r e s h a p e r . p y 
 
       ` ` ` 
 
 
 
 4 .   Y o u r   i m a g e   w i l l   b e   s t r e t c h e d   a n d   r e s i z e d .   I t   w i l l   a u t o m a t i c a l l y   s h o w   u p   a s   a   b a c k g r o u n d   i f   e n a b l e d   i n   t h e   G U I . 
 
 
 
 - - - 
 
 
 
 # #   D a t a   O v e r v i e w 
 
 
 
 d a t a _ l o a d e r . p y   h a n d l e s   t h r e e   d a t a s e t s : 
 
 
 
 -   T e m p e r a t u r e _ D a t a . c s v   -   h i s t o r i c a l   c l i m a t e   a n o m a l i e s     
 
 -   m e r g e d _ c o 2 _ t e m p . c s v   -   C O 2   a n d   t e m p e r a t u r e   c o m b i n e d   d a t a     
 
 -   g i s t e m p . c s v   -   l o n g - t e r m   N A S A   G I S   r e c o r d s     
 
 
 
 T h e s e   l i v e   i n   t h e   d a t a /   f o l d e r   a n d   a r e   n o r m a l i z e d   t o   l o w e r c a s e   f o r   c o l u m n   s a f e t y . 
 
 
 
 - - - 
 
 
 
 # #   N o t e s 
 
 
 
 -   A l l   v i s u a l i z a t i o n s   u s e   m a t p l o t l i b   w i t h   t h e   T k A g g   b a c k e n d   f o r   i n t e r a c t i v i t y .     
 
 -   I n s t a l l   d e p e n d e n c i e s   w i t h : 
 
 
 
     ` ` ` b a s h 
 
     p i p   i n s t a l l   n u m p y   s c i p y   s t a t s m o d e l s   m a t p l o t l i b   p a n d a s 
 
     ` ` ` 
 
 
 
 -   m a c O S   u s e r s :   y o u   m a y   n e e d   t o   i n v o k e   p y t h o n w   t o   r u n   t h e   G U I   p r o p e r l y .     
 
 -   F o r   r u b r i c   a n d   p l a n n i n g   d o c u m e n t s ,   s e e   t h e   p r o j e c t _ h i g h - l e v e l s /   f o l d e r . 
 
 
 
 - - - 
 
 
 
 E n j o y   e x p l o r i n g   c l i m a t e   t r e n d s   -   a n d   d o n ' t   f o r g e t   t o   b u g   J a s p e r   i f   t h i n g s   g e t   j a n k y ! 
 
 
 
 

---
subplot.py
---
#!/usr/bin/env python3
"""
subplot.py
----------
Tkinter GUI for interactive regression exploration.
After user closes the main plot, a historical reconstruction is shown.
"""

from __future__ import annotations

import numpy as np
from functools import partial
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["backend"] = "TkAgg"
import matplotlib.image as mpimg

import tkinter as tk
from tkinter import ttk, messagebox

import data_loader
from interactivity import attach_hover
from infer import long_term_inference


# ---------------- Fit helpers ------------------------------------------
def poly_fit(degree: int):
    def fit(x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exp_fit(x: np.ndarray, y: np.ndarray):
    x0 = x[0]

    def _model(x_vals, c, r, b):
        return c * np.exp(r * (x_vals - x0)) + b

    b0 = float(np.min(y))
    c0 = float(y[0] - b0)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=5000)
        return lambda x_new: _model(x_new, *params)
    except Exception as exc:
        print(f"[exp_fit] Fit failed with p0={p0}: {exc}")
        mean_y = float(np.mean(y))
        return lambda x_new: np.full_like(x_new, mean_y, dtype=float)


def loess_fit(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = smoothed[:, 0], smoothed[:, 1]
    return lambda x_new: np.interp(x_new, xs, ys)


# ---------------- Single-plot helper -----------------------------------
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
        attach_hover(
            ax,
            x,
            y,
            model,
            scatter_size=scatter_size,
            start=int(x.min()),
            end=future_end or last_x,
        )

    plt.show()
    return model


# ---------------- Tk GUI -----------------------------------------------
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

    # -------- UI builders & callbacks -----------------------------------
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

        fit_func = self.methods[self.method_var.get()]
        model = plot_regression(
            x,
            y,
            fit_func,
            title=f"{self.method_var.get()} Fit",
            xlabel="Year",
            ylabel="Temperature",
            scatter_size=50,
            future_end=self.future_var.get(),
            show_background=bool(self.bg_var.get()),
            interactive=bool(self.interactive_var.get()),
        )

        self._last_model = model
        self._last_x = x
        self._last_y = y
        self._last_show_bg = bool(self.bg_var.get())
        self._last_interactive = bool(self.interactive_var.get())

    # API for caller -----------------------------------------------------
    def get_last_fit(self):
        return (
            self._last_model,
            self._last_x,
            self._last_y,
            self._last_show_bg,
            self._last_interactive,
        )


# ---------------- Script entry-point ------------------------------------
def run():
    app = RegressionApp()
    app.mainloop()

    model, x_obs, y_obs, show_bg, interactive = app.get_last_fit()
    if model is None:
        print("No plot was generated â€“ nothing to infer. Exiting.")
        return

    print("Running historical inference ...")
    x_long, y_long = data_loader.load_long_data()
    fig, ax = long_term_inference(
        model,
        x_long,
        y_long,
        start_year=1000,
        end_year=1950,
        title="Historical reconstruction (1000-1950)",
        interactive=interactive,
    )

    if show_bg:
        try:
            bg = mpimg.imread("static/example2.png")
            x0, _ = ax.get_xlim()
            y0_existing, _ = ax.get_ylim()
            y0 = min(y0_existing, float(y_long.min()), float(y_obs.min()))
            ax.set_ylim(y0, float(np.max([y_long.max(), y_obs.max()])))
            ax.imshow(bg, aspect="auto", extent=(x0, x_obs.max(), y0, ax.get_ylim()[1]), zorder=0, alpha=1)
        except Exception as exc:
            print(f"[run] Warning: could not load background: {exc}")

    ax.scatter(x_long, y_long, s=25, c="blue", label="Long-Term Temperature (Obs.)", zorder=4)
    ax.scatter(x_obs, y_obs, s=25, c="blue", alpha=0.5, zorder=4)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run()


---
