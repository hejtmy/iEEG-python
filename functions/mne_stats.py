from scipy.stats import ttest_1samp, wilcoxon, ranksums, ttest_ind
import numpy as np
from matplotlib import gridspec
from matplotlib import colors
import matplotlib.pyplot as plt
from functions import mne_helpers as mnehelp
from functions import helpers
from functions import mne_plot_helpers as mneplothelp


# TFR is event X electrode X freqs X time
# Returns table and sued frequencies for laters
# Wilcox table is Channel X time X freqs
def wilcox_tfr_power(tfr1_orig, tfr2_orig, picks = None, average = False):
    tfr1 = tfr1_orig.copy()
    tfr2 = tfr2_orig.copy()
    if picks is not None:
        tfr1.pick_channels(picks)
        tfr2.pick_channels(picks)
    # reverse creates channel x freq X time X trials
    if average:
         tfr1.data = tfr1.data.mean(-3, keepdims = True)
         tfr2.data = tfr1.data.mean(-3, keepdims = True)
    events1 = mnehelp.reverse_tfr_list(tfr1.data)
    events2 = mnehelp.reverse_tfr_list(tfr2.data)
    freqs = tfr2.freqs
    
    wilcox_table = mnehelp.instantiate_tfr_zero_list(events1)
    print("=" * len(wilcox_table))
    for i_ch, channel in enumerate(events1):
        print("=", end = "")
        for i_freq, frequency in enumerate(channel):
            for i_time, timestamp in enumerate(frequency):
                time_res = ranksums(events1[i_ch][i_freq][i_time], events2[i_ch][i_freq][i_time])
                # returns positive p values and negative p values
                sign = 1 if time_res.statistic > 0 else -1
                wilcox_table[i_ch][i_freq][i_time] = time_res.pvalue * sign  # converts statistic to -+ 1 depending on the direction
    wilcox_table = np.asarray(wilcox_table)
    return wilcox_table, freqs


## PLOT ----------
# Wilcox table is in createschannel X time x freq
def plot_wilcox(wilcox_table, channel, sampling_frequency, cutout = 0.05, freqs = []):
    if len(freqs) == 0:
        freqs = range(len(wilcox_table[channel]))
    #recalculates to seconds from sampling freq
    times = range(len(wilcox_table[channel][0]))
    times = [time/sampling_frequency for time in times]
    x, y = np.meshgrid(times, freqs)
    
    p_values = np.array(wilcox_table[channel])
    p_values[p_values > cutout] = 1
    p_values[p_values < -cutout] = -1
    p_values[(0 < p_values) & (p_values <= cutout)] = 0.5
    p_values[(-cutout <= p_values) & (p_values < 0)] = -0.5
    plt.figure()
    plt.pcolormesh(x, y, p_values)
    plt.colorbar()
    plt.show()
    
    
# Explanation
def plot_wilcox_box(wilcox_table, sampling_frequency, cutout = 0.05, freqs = [], picks = [], pick_names = []):
    if len(freqs) == 0:
        freqs = range(len(wilcox_table[0]) + 1)
    if len(picks) == 0:
        picks = range(len(wilcox_table))
    # find the frequency index
    times = range(len(wilcox_table[0][0]))
    times = [time/sampling_frequency for time in times]
    x, y = np.meshgrid(times, freqs)
    
    nrow, ncol = helpers.nrow_ncol(len(picks))
    gs = gridspec.GridSpec(nrow, ncol)
    fig = plt.figure()
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap,norm = mneplothelp.significance_plot_norms(cutout)
    for i, channel in enumerate(picks):
        ax = fig.add_subplot(gs[i])
        #recalculates to seconds from sampling freq
        p_values = np.array(wilcox_table[i])
        p_values[p_values > cutout] = 1
        p_values[p_values < -cutout] = -1
        p_values[(p_values > 0) & (p_values <= cutout)] = cutout-0.0001#because of how >< works
        p_values[(p_values < 0) & (p_values >= -cutout)] = -cutout+0.0001
        im = ax.pcolormesh(x, y, p_values, cmap=cmap, norm=norm)
        ax.set_title([mnehelp.create_pick_name(i, pick_names)])
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax = cbar_ax)
    plt.show()
    