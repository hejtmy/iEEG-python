import matplotlib.pyplot as plt
import numpy as np


def plot_power_heatmap(tfr, ncol=6):
    """Plots 

    basically replaces plot topo power which for some reason I am quite unable to run 

    Parameters
    ----------
    tfr : [type]
        [description]
    ncol : int, optional
        [description], by default 6
    """
    nrow = int(np.ceil(len(tfr.ch_names)/ncol))
    if(nrow <= 1):
        Warning("Need to fix this for a single row")
        return
    fig, axs = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=True)
    # TODO - add 0 plotting
    x, xlabels = (0, len(tfr.times)), (tfr.times[0], tfr.times[-1])
    y, ylabels = range(0, len(tfr.freqs)), np.round(tfr.freqs, 3)
    for r in range(0, nrow):
        for c in range(0, ncol):
            i = c + r*ncol
            if i >= tfr.data.shape[0]:
                break
            im = axs[r, c].pcolormesh(tfr.data[i, ...])
            _ = axs[r, c].set_title(tfr.ch_names[i])
            _ = axs[r, c].set_xticks(x)
            _ = axs[r, c].set_xticklabels(xlabels)
            _ = axs[r, c].set_yticks(y)
            _ = axs[r, c].set_yticklabels(ylabels)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
