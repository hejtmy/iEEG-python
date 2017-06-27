from scipy.stats import ttest_1samp, wilcoxon, ranksums, ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

# TFR is event X electrode X freqs X time
def wilcox_tfr_power(tfr1, tfr2):
    events1 = reverse_tfr_list(tfr1.data)
    events2 = reverse_tfr_list(tfr2.data)
    wilcox_table = instantiate_zero_list(events1)
    for i_ch, channel in enumerate(events1):
        for i_freq, frequency in enumerate(channel):
            for i_time, timestamp in enumerate(frequency):
                wilcox_table[i_ch][i_freq][i_time] = ranksums(events1[i_ch][i_freq][i_time], events2[i_ch][i_freq][i_time]).pvalue
    
    
# reverses to Channel X Frequency X Time X Events
def reverse_tfr_list(ls):
    rev_ls = np.swapaxes(ls, 0, 3)
    rev_ls = np.swapaxes(rev_ls, 0, 2)
    rev_ls = np.swapaxes(rev_ls, 1, 0)
    return(rev_ls)


def instantiate_zero_list(ls):
    a, b, c = len(ls), len(ls[0]), len(ls[0][0])
    zeros = [[[0 for x in range(c)] for y in range(b)] for z in range(a)]
    return(zeros)


def plot_wilcox(wilcox_table, channel, sampling_frequency):
    freqs = range(len(wilcox_table[channel]))
    times = range(len(wilcox_table[channel][0]))
    times = [time/sampling_frequency for time in times]
    x, y = np.meshgrid(times, freqs)
    p_values = np.array(wilcox_table[channel])
    p_values[p_values > 0.1] = 1
    plt.figure()
    plt.pcolormesh(x, y, p_values)
    plt.colorbar()
    plt.show()
