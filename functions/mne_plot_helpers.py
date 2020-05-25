from matplotlib import colors, gridspec
from matplotlib import pyplot as plt
import numpy as np
from functions import helpers

from mne import time_frequency as tfr


def significance_plot_norms(cutout=0.05):
    cmap = colors.ListedColormap(['blue', 'red', 'yellow', 'blue'])
    bounds = [-1, -cutout, 0, cutout, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# requires non averaged EpochsTFR object
def plot_epochs_power(epochs, frequency, picks=None):
    epochsTFR = epochs.copy()
    if not isinstance(epochsTFR, tfr.EpochsTFR):
        print('Not Epochs TFR object')
        return
    if picks is not None:
        epochsTFR.pick_channels(picks)
    # recalculates to seconds from sampling freq
    n_epochs = epochsTFR.data.shape[0]
    x, y = np.meshgrid(epochsTFR.times, range(n_epochs))

    nrow, ncol = helpers.nrow_ncol(epochsTFR.data.shape[1])
    gs = gridspec.GridSpec(nrow, ncol)
    fig = plt.figure()
    for i, channel in enumerate(epochsTFR.ch_names):
        ax = fig.add_subplot(gs[i])
        # recalculates to seconds from sampling freq
        values = np.array(epochsTFR.data[:, i, frequency, :])
        im = ax.pcolormesh(x, y, values)
        ax.set_title(epochsTFR.ch_names[i])
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im)
    plt.show()
