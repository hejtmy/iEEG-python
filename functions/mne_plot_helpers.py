from matplotlib import colors 

def significance_plot_norms(cutout = 0.05):
    cmap = colors.ListedColormap(['blue', 'red', 'yellow', 'blue'])
    bounds = [-1, -cutout, 0, cutout, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm