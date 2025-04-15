import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk

def regression_plot(xvar, yvar, mhat, bhat, r, r2):
    """
    Scatterplot the data and overlay the least‐squares regression line using the cyberpunk theme.
    """
    print(f"  slope (m)     = {mhat:.4f}")
    print(f"  intercept (b) = {bhat:.4f}")
    print(f"  R²            = {r2:.4f}")

    x_line = np.array([xvar.min(), xvar.max()])
    y_line = mhat * x_line + bhat

    fig, ax = plt.subplots()
    plt.style.use("cyberpunk")
    ax.set_facecolor('#1f1f1f')
    fig.patch.set_facecolor('#1f1f1f')

    ax.scatter([xvar[0]], [yvar[0]], color='#00ffff', s=40, edgecolors='none', alpha=0)
    ax.scatter(xvar[1:], yvar[1:], color='#00ffff', s=40, edgecolors='none')
    ax.plot(x_line, y_line, color='#ff00ff', linewidth=2)
    mplcyberpunk.add_glow_effects()

    ax.set_xlabel("total atmospheric CO2, ppm", color='white')
    ax.set_ylabel("global temperature", color='white')
    ax.set_title("Regression Plot (cyberpunk)", color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    return fig, ax
