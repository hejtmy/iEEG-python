import mne
from matplotlib import gridspec
from mne.channels.layout import Layout
from functions import helpers
import numpy as np
import matplotlib.pyplot as plt


def picks_all_localised(raw, pd_montage, where):
    all_channels = picks_all(raw)
    where_channels = picks_localised(pd_montage, where)
    return np.intersect1d(all_channels, where_channels)


def picks_all(raw, where=None, pd_montage=None):
    return mne.pick_types(raw.info, seeg=True, meg=True, eeg=False,
                          stim=False, eog=True, exclude='bads')


def picks_localised(pd_montage, name):
    return pd_montage[pd_montage.neurologyLabel.str.contains(name)].index


# Droping epochs helpers
def get_dropped_epoch_indices(ls):
    rm_ignored = [item for item in ls if item != ['IGNORED']]
    drop_indices = [i for i, x in enumerate(rm_ignored) if x == ['USER']]
    return(drop_indices)


# Creates box layout for non standard topological images (such as seeg)
# NEEDS channel names, unfortunately
def custom_box_layout(ch_names, ncol=6):
    # Creates boundaries of the 0-1 box, will be
    box = (-0.1, 1.1, -0.1, 1.1)
    # just puts the channel names to strings
    nrow, ncol = helpers.nrow_ncol(len(ch_names), ncol=ncol)
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


# frequency is in indes at this time
# picks are in indices of already computed
def plot_power_time(tfrs, picks, frequency, pick_names=[],
                    event_names=[], ncol=3):
    # find the frequency index
    nplots = len(picks) + 1  # last will be legend
    nrow, ncol = helpers.nrow_ncol(nplots, ncol)
    gs = gridspec.GridSpec(nrow, ncol)
    fig = plt.figure()
    for i, pick in enumerate(picks):
        ax = fig.add_subplot(gs[i])
        for n, event_tfr in enumerate(tfrs):
            ax.plot(event_tfr.times, event_tfr.data[pick, frequency, :],
                    label=event_names[n])
            ax.legend([create_pick_name(i, pick_names)])
    # adds legend
    ax = fig.add_subplot(gs[nplots - 1])
    for n, _ in enumerate(tfrs):
        ax.plot([0], label=event_names[n])
        handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.show()


def plot_power_time_average(tfrs, picks, frequency, channel_name=[], event_names=[], ncol=3):
    # find the frequency index
    nplots = len(picks) + 1  # last will be legend
    nrow, ncol = helpers.nrow_ncol(nplots, ncol)
    plt.figure()
    ax = plt.axes()
    for n, event_tfr in enumerate(tfrs):
        data = event_tfr.data[picks, frequency, :]
        data = np.average(data, 0).squeeze()
        ax.plot(event_tfr.times, data, label=event_names[n])
    ax.legend(channel_name)
    plt.legend()
    plt.show()


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
                axarr[i, idx] = event_epoch.plot_psd(fmin = fmin, fmax = fmax,
                          picks = [channel], color = color, show = False)
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


# reverses to Channel X Frequency X Time X Events
def reverse_tfr_list(ls):
    rev_ls = np.swapaxes(ls, 0, 3)
    rev_ls = np.swapaxes(rev_ls, 0, 2)
    rev_ls = np.swapaxes(rev_ls, 1, 0)
    return(rev_ls)


# returns 0s given by length of first three elements of list for Wilcox Channel X Frequency X time
def instantiate_tfr_zero_list(ls):
    a, b, c = len(ls), len(ls[0]), len(ls[0][0])
    zeros = [[[0 for x in range(c)] for y in range(b)] for z in range(a)]
    return(zeros)
