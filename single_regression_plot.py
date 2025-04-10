import numpy as np
import matplotlib.pyplot as plt

# Attempt to import mplcyberpunk for the Cyberpunk theme.
try:
    import mplcyberpunk
except ImportError:
    mplcyberpunk = None


def regression_plot(xvar, yvar, theme=1):
    """
    Displays a regression line plot of xvar vs yvar using one of five themes.

    Themes:
      1. Cyberpunk: Dark background using mplcyberpunk with white text.
      2. Wild West: Earthy tan backdrop with a brown line and 'x' markers.
      3. Minimalist: White background, black line, minimal ticks only at data limits, no extra spines or grid.
      4. Hellish: Deep red background with a bright red line, orange markers, dark red gridlines, and yellow text.
      5. Angelic: White background, sky blue line, pale gridlines and slightly darker blue text.
    """
    plt.figure()
    ax = plt.gca()

    if theme == 1:
        # Cyberpunk theme
        if mplcyberpunk is None:
            print("mplcyberpunk is not installed. Please install it to use the Cyberpunk theme.")
            return
        plt.style.use("cyberpunk")
        # Force a dark background for axes and overall figure.
        ax.set_facecolor('#1f1f1f')
        plt.gcf().patch.set_facecolor('#1f1f1f')
        # Plot the data
        ax.plot(xvar, yvar, color='#00ffff', linewidth=2)
        ax.grid(True, linestyle="--", linewidth=1, color='gray')
        mplcyberpunk.add_glow_effects()  # Add glowing effects after plotting.
        # Set all text elements to white.
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
    elif theme == 2:
        # Wild West theme: Set an earthy tan backdrop.
        ax.set_facecolor('#f4e2d8')
        plt.gcf().patch.set_facecolor('#f4e2d8')
        ax.plot(xvar, yvar, color='#6b4423', marker='x', linestyle='-', linewidth=2)
        ax.grid(True, linestyle='-', linewidth=1, color='#c2b280')
    elif theme == 3:
        # Minimalist theme: white background, clean look, and minimal ticks only at data limits.
        ax.set_facecolor('white')
        plt.gcf().patch.set_facecolor('white')
        ax.plot(xvar, yvar, color='black', linewidth=2)
        ax.grid(False)
        # Remove all spines.
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Set only ticks at the minimum and maximum values.
        xticks = [np.min(xvar), np.max(xvar)]
        yticks = [np.min(yvar), np.max(yvar)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        # Use a default font available in matplotlib to avoid Helvetica issues.
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
    elif theme == 4:
        # Hellish theme: deep red background with bright red line and orange markers.
        ax.set_facecolor('#1a0000')
        plt.gcf().patch.set_facecolor('#1a0000')
        ax.plot(xvar, yvar, color='red', marker='o', linestyle='-', linewidth=2, markerfacecolor='orange')
        ax.grid(True, linestyle="--", linewidth=1, color='#8B0000')
        # Set all text elements to yellow.
        ax.title.set_color('yellow')
        ax.xaxis.label.set_color('yellow')
        ax.yaxis.label.set_color('yellow')
        ax.tick_params(colors='yellow')
    elif theme == 5:
        # Angelic theme: white background, sky blue line, pale gridlines, and darker blue text.
        ax.set_facecolor('white')
        plt.gcf().patch.set_facecolor('white')
        ax.plot(xvar, yvar, color='#87ceeb', linewidth=2)
        ax.grid(True, linestyle="--", linewidth=1, color='#e0f7fa')
        # Set text colors.
        ax.title.set_color('#4682b4')
        ax.xaxis.label.set_color('#4682b4')
        ax.yaxis.label.set_color('#4682b4')
        ax.tick_params(colors='#4682b4')
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
    else:
        # Fallback to default plotting.
        ax.plot(xvar, yvar, color='black', linewidth=2)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Regression Plot")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    theme_input = input("Select a theme (1: Cyberpunk, 2: Wild West, 3: Minimalist, 4: Hellish, 5: Angelic): ")
    try:
        theme = int(theme_input)
        if theme not in [1, 2, 3, 4, 5]:
            print("Invalid theme selected. Defaulting to theme 1 (Cyberpunk).")
            theme = 1
    except ValueError:
        print("Invalid input. Defaulting to theme 1 (Cyberpunk).")
        theme = 1

    # Create a sample dataset.
    x = np.linspace(0, 10, 100)
    y = np.sin(x)  # Sample regression data

    # Plot the regression with the chosen theme.
    regression_plot(x, y, theme)
