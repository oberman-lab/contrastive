import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_requires_grad(m, requires_grad):
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(requires_grad)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(requires_grad)


class RunningAvg:
    def __init__(self):
        self.mean = 0
        self.N = 0

    def add(self, x):
        N = self.N
        self.mean = (self.mean * N + x) / (N + 1)
        self.N = N + 1

    def get(self):
        return self.mean

    def wipe(self):
        self.mean = 0
        self.N = 0


def plot_by_categories(x, y, labels, weight_vectors, num_classes, num_points):
    # setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # define the data
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, num_classes, num_classes + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(x, y, c=labels, s=2000 / num_points, cmap=cmap, norm=norm)
    ax.scatter(weight_vectors[0], weight_vectors[1],s=100,c="black", marker="P")
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    return fig