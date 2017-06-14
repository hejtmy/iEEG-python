import mne
from mne.channels.layout import Layout
from functions import helpers
import numpy as np
import matplotlib.pyplot as plt

def picks_all_localised(raw, pd_montage, where):
    all_channels = picks_all(raw)
    where_channels = picks_localised(pd_montage, where)
    return np.intersect1d(all_channels, where_channels)

def picks_all(raw, where = None, pd_montage = None):
    picks = mne.pick_types(raw.info, seeg = True, meg = True, eeg = False, stim = False, eog = True, exclude ='bads')
    return picks

def picks_localised(pd_montage, name):
    return pd_montage[name == pd_montage.neurologyLabel].index

# Creates box layout for non standard topological images (such as seeg)
# NEEDS channel names, unfortunately
def custom_box_layout(ch_names, ncol = 6):
    # Creates boundaries of the 0-1 box, will be
    box = (-0.1, 1.1, -0.1, 1.1)
    # just puts the channel names to strings
    nrow = np.ceil(len(ch_names) / ncol)
    x, y = np.mgrid[0:1:(1/ncol), 0:1:(1/nrow)]
    xy = np.vstack([x.ravel(), y.ravel()]).T
    w, h = [1 / (1.1 * ncol), 1 / (1.1 * nrow)]
    w, h = [np.tile(i, xy.shape[0]) for i in [w, h]]
    pos = np.hstack([xy, w[:, None], h[:, None]])
    # removes redundat column
    pos = pos[:len(ch_names)]
    ch_indices = range(len(ch_names))
    box_layout = Layout(box, pos, ch_names, ch_indices, 'box')
    return(box_layout)


