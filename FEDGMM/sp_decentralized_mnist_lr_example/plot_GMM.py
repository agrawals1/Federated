import numpy as np
import matplotlib.pyplot as plt
class PlotElement:
    def __init__(self, x, y, title, normalize=False):
        self.x = x
        self.y = y
        self.title = title
        self.normalize = normalize

    def plot(self, ax=None, save_path=None):
        if ax is None:
            fig, ax = plt.subplots()

        if self.normalize:
            x_data = self.x / self.x.max()
            y_data = self.y / self.y.max()
        else:
            x_data = self.x
            y_data = self.y

        # Use different markers for each plot
        marker = {'Predicted Causal Effect (Ours)': 'o',  # Circle
                  'Actual Causal Effect': 'v',  # Triangle Down
                  'Direct predictions from Neural Network': 's'}  # Square

        # Plot data with specified styles
        ax.plot(x_data, y_data, label=self.title, marker=marker[self.title],
                markersize=2, linewidth=2)  # Increased markersize and linewidth
        ax.set_title("Comparison of Model Predictions")
        ax.legend(loc='upper left', fontsize='small', framealpha=0.8, handlelength=2)  # Adjusted fontsize

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return ax


