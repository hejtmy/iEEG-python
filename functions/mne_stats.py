from scipy.stats import ttest_1samp, wilcoxon, ranksums, ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

# TFR is event X electrode X freqs X time
def wilcox_tfr_power(tfr1, tfr2):
    events1 = reverse_tfr_list(tfr1.data)
    events2 = reverse_tfr_list(tfr2.data)
    freqs = tfr2.freqs
    # reverse creates channel x freq X time X trials
    wilcox_table = instantiate_zero_list(events1)
    print("=" * len(wilcox_table))
    for i_ch, channel in enumerate(events1):
        print("=", end = "")
        for i_freq, frequency in enumerate(channel):
            for i_time, timestamp in enumerate(frequency):
                time_res = ranksums(events1[i_ch][i_freq][i_time], events2[i_ch][i_freq][i_time])
                # returns positive p values and negative p values
                wilcox_table[i_ch][i_freq][i_time] = time_res.pvalue * (time_res.statistic/abs(time_res.statistic)) # converts statistic to -+ 1 depending on the direction
    return wilcox_table, freqs
    

# reverses to Channel X Frequency X Time X Events
def reverse_tfr_list(ls):
    rev_ls = np.swapaxes(ls, 0, 3)
    rev_ls = np.swapaxes(rev_ls, 0, 2)
    rev_ls = np.swapaxes(rev_ls, 1, 0)
    return(rev_ls)

    
# returns 0s given by length of first three elements of list for Wilcox Channel X Frequency X time
def instantiate_zero_list(ls):
    a, b, c = len(ls), len(ls[0]), len(ls[0][0])
    zeros = [[[0 for x in range(c)] for y in range(b)] for z in range(a)]
    return(zeros)


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