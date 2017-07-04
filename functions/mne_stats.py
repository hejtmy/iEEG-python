from scipy.stats import ttest_1samp, wilcoxon, ranksums, ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
from functions import mne_helpers as mnehelp
from functions import helpers

# TFR is event X electrode X freqs X time
# Returns table and sued frequencies for laters
# Wilcox table is Channel X time X freqs
def wilcox_tfr_power(tfr1, tfr2):
    events1 = mnehelp.reverse_tfr_list(tfr1.data)
    events2 = mnehelp.reverse_tfr_list(tfr2.data)
    freqs = tfr2.freqs
    # reverse creates channel x freq X time X trials
    wilcox_table = mnehelp.instantiate_tfr_zero_list(events1)
    print("=" * len(wilcox_table))
    for i_ch, channel in enumerate(events1):
        print("=", end = "")
        for i_freq, frequency in enumerate(channel):
            for i_time, timestamp in enumerate(frequency):
                time_res = ranksums(events1[i_ch][i_freq][i_time], events2[i_ch][i_freq][i_time])
                # returns positive p values and negative p values
                wilcox_table[i_ch][i_freq][i_time] = time_res.pvalue * (time_res.statistic/abs(time_res.statistic)) # converts statistic to -+ 1 depending on the direction
    wilcox_table = np.asarray(wilcox_table)
    return wilcox_table, freqs


# Wilcox table is in createschannel X time x freq
def plot_wilcox(wilcox_table, channel, sampling_frequency, cutout = 0.05, freqs = []):
    if len(freqs) == 0:
        freqs = range(len(wilcox_table[channel]))
    #recalculates to seconds from sampling freq
    times = range(len(wilcox_table[channel][0]))
    times = [time/sampling_frequency for time in times]
    x, y = np.meshgrid(times, freqs)
    
    p_values = np.array(wilcox_table[channel])
    p_values[abs(p_values) > cutout] = 1
    p_values[(0 < p_values) & (p_values <= cutout)] = 0.5
    p_values[(cutout <= p_values) & (p_values < 0)] = -0.5
    plt.figure()
    plt.pcolormesh(x, y, p_values)
    plt.colorbar()
    plt.show()
    
    
# Explanation
def plot_wilcox_box(wilcox_table, sampling_frequency, cutout = 0.05, freqs = [], channels = []):
    if len(freqs) == 0:
        freqs = range(len(wilcox_table[0]) + 1)
    if len(channels) == 0:
        channels = range(len(wilcox_table))
    times = range(len(wilcox_table[0][0]))
    times = [time/sampling_frequency for time in times]
    x, y = np.meshgrid(times, freqs)
    nrow, ncol = helpers.nrow_ncol(len(channels))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    for i_channel in channels:
        #recalculates to seconds from sampling freq
        p_values = np.array(wilcox_table[i_channel])
        p_values[abs(p_values) > cutout] = 1
        p_values[(0 < p_values) & (p_values <= cutout)] = 0.5
        p_values[(cutout <= p_values) & (p_values < 0)] = -0.5
        i_row, i_col = helpers.layout_position(i_channel, nrow, ncol)
        axes[i_row, i_col].pcolormesh(x, y, p_values)
    plt.show()
