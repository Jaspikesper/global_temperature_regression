import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcyberpunk

def regression_plot(xvar, yvar, theme=1):
    """
    Scatter the data and overlay the least‐squares regression line.
    Themes (by index):
      1. cyberpunk
      2. wild west
      3. minimalist
      4. communist
      5. angelic
    """
    # 1. Compute slope, intercept, and R²
    mhat = np.cov(xvar, yvar)[0, 1] / np.var(xvar)
    bhat = np.mean(yvar) - mhat * np.mean(xvar)
    r = np.corrcoef(xvar, yvar)[0, 1]
    r2 = r**2

    # 2. Print the regression info
    print(f"  slope (m)     = {mhat:.4f}")
    print(f"  intercept (b) = {bhat:.4f}")
    print(f"  R²            = {r2:.4f}")

    # 3. Build the line over the data range
    x_line = np.array([xvar.min(), xvar.max()])
    y_line = mhat * x_line + bhat

    # 4. Plot
    fig, ax = plt.subplots()

    # pick a human‑readable name
    theme_names = ["cyberpunk", "wild west", "minimalist", "communist", "angelic"]
    theme_name = theme_names[theme-1] if 1 <= theme <= 5 else str(theme)

    if theme == 1:
        # cyberpunk
        if mplcyberpunk is None:
            print("Install mplcyberpunk for the Cyberpunk theme.")
            return
        plt.style.use("cyberpunk")
        ax.set_facecolor('#1f1f1f')
        fig.patch.set_facecolor('#1f1f1f')

        ax.scatter(xvar, yvar, color='#00ffff', s=40, edgecolors='none')
        ax.plot(x_line, y_line, color='#ff00ff', linewidth=2)
        mplcyberpunk.add_glow_effects()
        text_color = 'white'

    elif theme == 2:
        # wild west
        ax.set_facecolor('#f4e2d8')
        fig.patch.set_facecolor('#f4e2d8')

        ax.scatter(xvar, yvar, color='#6b4423', marker='x', s=50)
        ax.plot(x_line, y_line, color='#6b4423', linewidth=2)
        ax.grid(True, linestyle='-', linewidth=1, color='#c2b280')
        text_color = 'black'

    elif theme == 3:
        # minimalist
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        ax.scatter(xvar, yvar, color='black', s=30)
        ax.plot(x_line, y_line, color='black', linewidth=2)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([x_line[0], x_line[1]])
        ax.set_yticks([y_line[0], y_line[1]])
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
        text_color = 'black'

    elif theme == 4:
        # communist
        ax.set_facecolor('#1a0000')
        fig.patch.set_facecolor('#1a0000')

        ax.scatter(xvar, yvar, color='red', marker='o', s=50, edgecolors='orange')
        ax.plot(x_line, y_line, color='red', linewidth=2)
        ax.grid(True, linestyle="--", linewidth=1, color='#8B0000')
        text_color = 'yellow'

    elif theme == 5:
        # angelic
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        ax.scatter(xvar, yvar, color='#87ceeb', s=40)
        ax.plot(x_line, y_line, color='#4682b4', linewidth=2)
        ax.grid(True, linestyle="--", linewidth=1, color='#e0f7fa')
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
        text_color = '#4682b4'

    else:
        # fallback
        ax.scatter(xvar, yvar, color='black', s=30)
        ax.plot(x_line, y_line, color='black', linewidth=2)
        text_color = 'black'

    # common labels (updated)
    ax.set_xlabel("total atmospheric CO2, ppm", color=text_color)
    ax.set_ylabel("global temperature", color=text_color)
    ax.set_title(f"Regression Plot ({theme_name})", color=text_color)
    ax.tick_params(colors=text_color)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # demo with your CSV
    data = pd.read_csv('merged_co2_temp.csv')
    x = data['mean'].values
    y = data['temp_anomaly'].values
    regression_plot(x, y, theme=1)
