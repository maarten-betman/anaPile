import matplotlib.pyplot as plt


class BasePlot:
    def __init__(self):
        self.fig = None

    def _create_fig(self, figsize, dpi=100):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        return self.fig

    def _finish_plot(self, grid=False, show=True):
        if show:
            plt.show()
        if grid:
            plt.grid()
        return self.fig
