import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['backend'] = 'TkAgg'  # Sets the interactive backend (verify if needed for your environment)
from data_loader import x, y


def plot_interactive_regression(x, y, fit_func, background_img_path=None,
                                title="Interactive Regression Plot",
                                xlabel="Year",
                                ylabel="y",
                                scatter_label="Data",
                                regression_label="Regression Fit",
                                scatter_kwargs=None,
                                line_kwargs=None,
                                scatter_size=2):
    """
    Creates an interactive regression plot with an optional background image.

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        fit_func (callable): A function that, given x and y, returns a regression model.
                             The model must be callable such that model(x) returns prediction(s).
        background_img_path (str, optional): File path to an image to use as the plot background.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        scatter_label (str): The label used for the scatter data in the legend.
        regression_label (str): The label used for the regression line in the legend.
        scatter_kwargs (dict, optional): Additional keyword arguments for the scatter plot.
        line_kwargs (dict, optional): Additional keyword arguments for the regression line plot.
        scatter_size (int): Size of the scatter plot markers.

    Returns:
        fig (Figure): The created matplotlib figure.
        ax (Axes): The axes for the primary plot.
        model (callable): The regression model produced by fit_func.

    Notes:
        - The function sets up an interactive hover event that updates an annotation with the
          regression prediction at the rounded x position of the cursor.
        - If 'x' values are not integers or do not represent years, the conversion using
          int(round(event.xdata)) in the hover event might require adjustment.
        - The background image (if provided) is added to a new axes placed behind the primary plot.
    """

    # Set default kwargs if None provided.
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    # Create the regression model using fit_func and compute predictions.
    model = fit_func(x, y)
    y_pred = model(x)

    # Create the new figure.
    fig = plt.figure()

    # Optionally load and set the background image.
    if background_img_path:
        try:
            bg_img = mpimg.imread(background_img_path)
            # Create an axes for the background image covering the whole figure.
            bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
            bg_ax.imshow(bg_img, aspect='auto')
            bg_ax.axis('off')
        except Exception as e:
            print(f"Error loading background image: {e}")
            # Note: If the image path is incorrect or image cannot be loaded, execution continues.

    # Create the primary axes with transparent background to overlay on the background image.
    ax = fig.add_axes([0.125, 0.11, 0.775, 0.77], zorder=1)
    ax.set_facecolor('none')

    # Prepare scatter plot options, ensuring that scatter_size takes precedence.
    scatter_defaults = {
        'marker': 'o',
        'c': 'black',
        'edgecolors': 'black',
        'alpha': 1.0,
        'zorder': 2
    }
    scatter_defaults.update(scatter_kwargs)
    scatter_defaults['s'] = scatter_size  # Set marker size explicitly.
    ax.scatter(x, y, label=scatter_label, **scatter_defaults)

    # Create an empty scatter for the halo highlight on hover (yellow circle).
    print(scatter_defaults['s'])
    halo_scatter = ax.scatter(np.empty((0, 2)), np.empty((0, 2)),
                               facecolors='none', edgecolors='yellow',
                              linewidths=2, zorder=4, s=scatter_defaults['s'])

    # Prepare and plot the regression line.
    line_defaults = {
        'color': 'orange',
        'linewidth': 2,
        'zorder': 3
    }
    line_defaults.update(line_kwargs)
    ax.plot(x, y_pred, label=regression_label, **line_defaults)

    # Set the title and axis labels with larger bold fonts.
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    # Increase tick label font size, set them bold, and enforce black color.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        label.set_color('black')

    # Create and customize the legend.
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_color('black')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Create an annotation that will be updated when hovering over the plot.
    hover_annotation = ax.annotate("",
                                   xy=(0, 0),
                                   xytext=(10, 10),
                                   textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w"),
                                   arrowprops=dict(arrowstyle="->", color='black'))
    hover_annotation.set_visible(False)

    def on_move(event):
        """
        Event handler for mouse movements. Updates the hover annotation and highlights the
        corresponding data point if the mouse is within the axes.
        """
        # Check if the mouse is inside the primary axes and if data coordinates are valid.
        if event.inaxes == ax and event.xdata is not None:
            # Convert the floating point x-coordinate to the nearest integer year.
            # Note: This conversion assumes the x values are integers (e.g., representing years).
            hovered_year = int(round(event.xdata))
            pred_y = model(hovered_year)

            # Update the annotation text and position.
            hover_annotation.set_text(f"Year = {hovered_year}\ny = {pred_y:.2f}")
            hover_annotation.xy = (hovered_year, pred_y)
            hover_annotation.set_visible(True)

            # Find indices in x matching the hovered_year exactly.
            indices = np.where(x == hovered_year)[0]
            if indices.size > 0:
                idx = indices[0]
                halo_scatter.set_offsets(np.array([[x[idx], y[idx]]]))
            else:
                # Clear halo highlight if no exact matching data point is found.
                halo_scatter.set_offsets(np.empty((0, 2)))
        else:
            # Hide the annotation and clear the halo highlight when mouse is out of bounds.
            hover_annotation.set_visible(False)
            halo_scatter.set_offsets(np.empty((0, 2)))

        fig.canvas.draw_idle()

    # Connect the mouse motion event to the on_move callback.
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Display the interactive plot.
    plt.show()

    return fig, ax, model


# --- Example usage ---
if __name__ == '__main__':
    np.random.seed(0)
    # Define a simple quadratic polynomial regression fit.
    quadratic_fit = lambda x, y: np.poly1d(np.polyfit(x, y, deg=2))

    fig, ax, model = plot_interactive_regression(
        x, y, quadratic_fit,
        background_img_path='example2.png',  # Update this path accordingly or set to None if not needed.
        title="Interactive Quadratic Regression",
        xlabel="Year",
        ylabel="y",
        scatter_label="Data",
        regression_label="Quadratic Fit",
        scatter_size=50  # Modify dot size as desired.
    )
