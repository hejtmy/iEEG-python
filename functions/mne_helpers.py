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
    return mne.pick_types(raw.info, seeg = True, meg = True, eeg = False, 
                           stim = False, eog = True, exclude ='bads')


def picks_localised(pd_montage, name):
    return pd_montage[name == pd_montage.neurologyLabel].index


##Droping epochs helpers
def get_dropped_epoch_indices(ls):
    rm_ignored = [item for item in ls if item != ['IGNORED']]
    drop_indices = [i for i, x in enumerate(rm_ignored) if x == ['USER']]
    return(drop_indices)

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


## PLOTS
def plot_psd_epochs_separate(epochs, picks, fmin, fmax, pick_names = [], event_names = []):
    plt.figure()
    ax = plt.axes()
    pick_colors = [helpers.random_matplotlib_color() for _ in range(len(picks))]
    event_linestyles = [helpers.int_to_linestyle(i) for i in range(len(epochs))]
    channel_names = []
    for i, event_epoch in enumerate(epochs):
        event_name = create_event_name(i, event_names)
        linestyle = event_linestyles[i]
        for idx, pick in enumerate(picks):
            pick_name = create_pick_name(idx, pick_names)
            color = pick_colors[idx]
            for channel in pick:
                event_epoch.plot_psd(fmin = fmin, fmax = fmax,
                          picks = [channel], color = color, show = False, ax = ax)
                ax.lines[-1].set_linestyle(linestyle)
                channel_name = "%s-%s: %s" % (pick_name, str(channel), event_name)
                channel_names.append(channel_name)
    ax.set_title('Plot')
    plt.legend(ax.lines, channel_names)
    plt.show()


def plot_psd_epochs_sep_box(epochs, picks, fmin, fmax, pick_names = [], event_names = []):
    f, axarr = plt.subplots(len(epochs), len(picks))
    ax = plt.axes()
    pick_colors = [helpers.random_matplotlib_color() for _ in range(len(picks))]
    event_linestyles = [helpers.int_to_linestyle(i) for i in range(len(epochs))]
    channel_names = []
    for i, event_epoch in enumerate(epochs):
        event_name = create_event_name(i, event_names)
        linestyle = event_linestyles[i]
        for idx, pick in enumerate(picks):
            pick_name = create_pick_name(idx, pick_names)
            color = pick_colors[idx]
            for channel in pick:
                event_epoch.plot_psd(fmin = fmin, fmax = fmax,
                          picks = [channel], color = color, show = False, ax = ax)
                ax.lines[-1].set_linestyle(linestyle)
                channel_name = "%s-%s: %s" % (pick_name, str(channel), event_name)
                channel_names.append(channel_name)
    ax.set_title('Plot')
    plt.legend(ax.lines, channel_names)
    plt.show()


def plot_psd_epochs(epochs, picks, fmin, fmax, tmin = [], tmax = [], pick_names = [], event_names = []):
    tmin, tmax = tmax_tmin_fill(epochs, tmin, tmax)
    
    plt.figure()
    ax = plt.axes()
    final_names = []
    pick_colors = [helpers.random_matplotlib_color() for _ in range(len(picks))]
    event_linestyles = [helpers.int_to_linestyle(i) for i in range(len(epochs))]
    for i, event_epoch in enumerate(epochs):
        event_name = create_event_name(i, event_names)
        for idx, pick in enumerate(picks):
            event_epoch.plot_psd(fmin = fmin, fmax = fmax,
                          picks = pick, color = pick_colors[idx],
                          show = False, ax = ax)
            ax.lines[-1].set_linestyle(event_linestyles[i])
            # creates line name
            pick_name = create_pick_name(idx, pick_names)
            final_names.append(pick_name + ": " + event_name)
    ax.set_title('Plot')
    plt.legend(ax.lines, final_names)
    plt.show()


def create_event_name(i, event_names):
    if (len(event_names) - 1 >= i):
        return event_names[i]
    else: 
        return 'event' + str(i)


def create_pick_name(i, pick_names):
    if (len(pick_names) - 1 >= i):
        return pick_names[i]
    else: 
        return 'electrodes' + str(i)
    
def tmax_tmin_fill(epochs, tmin, tmax):
    if tmax == []:
        tmax = epochs.tmax
    if tmin == []:
        tmin = epochs.tmin
    return tmin, tmax