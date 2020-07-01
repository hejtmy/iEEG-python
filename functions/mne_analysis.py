import numpy as np
from functions import mne_stats as mnestats
import mne
import scipy.stats as stats


# ANALYSIS -------
def compare_events_averaged_power(tfr, event1_name, event2_name,
                                  time=None, method='ttest'):
    """Calculates difference in power averaged over time between two events

    Calculates differences in averaged power in given frequency and on given
    channel. Returns statistics of given tests and their pvalues

    Parameters
    ----------
    tfr : mne.EpochsTFR
        [description]
    event1_name : str
        [description]
    event2_name : str
        [description]
    time : touple(float, float), optional
        if set, averages only given timeframe. If none, calculates on the
        entire epoch span, by default None
    method : str, optional
        [description], by default 'ttest'

    Returns
    -------
    stats, pvalues : np.ndarray, np.ndarray
        stats and pvalues as obtained by the test. Results have
        channel x frequency character. E.g. [0, 2] will have
        results for the first channel and third frequency
    """

    data = tfr.copy()
    if time is not None:
        data.crop(*time)
    average_event1 = np.average(data[event1_name].data, axis=-1)
    average_event2 = np.average(data[event2_name].data, axis=-1)
    # Epochs are epoch x channels x freqs x timen averaged over time
    n_channels = average_event1.shape[1]
    n_frequencies = average_event1.shape[2]
    
    statistics = np.empty(average_event1.shape[1:3])
    pvalues = np.empty(statistics.shape)
    for iChannel in range(0, n_channels):
        for iFrequency in range(0, n_frequencies):
            tempStat, tempP = stats.ttest_ind(
                average_event1[:, iChannel, iFrequency],
                average_event2[:, iChannel, iFrequency],
                equal_var=False)
            statistics[iChannel, iFrequency] = tempStat
            pvalues[iChannel, iFrequency] = tempP
    return statistics, pvalues


# PREPROCESSING -------
def band_power(tfr, bands):
    """Averages TFR object into given bands

    Parameters
    ----------
    tfr : mne.AverageTFR or mne.EpochsTFR
        object to average, passed by refernece
    bands : list of num touples
        list of  of both bottom and top inclusive touples of wanted bands.
        e.g. [(2, 4),(4, 9)] will select bands [2, 4] and [4, 9]
    Returns
    -------
    Returns the same object as was passed but with recalculated powers
        [description]

    Example
    -------
    """
    valid = False
    if isinstance(tfr, mne.time_frequency.tfr.EpochsTFR):
        # EpochsTFR are (n_epochs, n_channels, n_freqs, n_times)
        dim_freq = 2
        valid = True
    if isinstance(tfr, mne.time_frequency.tfr.AverageTFR):
        # AverageTFR are (n_channels, n_freqs, n_times)
        dim_freq = 1
        valid = True
    if not valid:
        print("This doesn't look like a TFR file. Dimensions are not correct")
        return None
    new_data = []
    new_freqs = []
    for band in bands:
        # finds indices of these freqs
        # TODO - add output of this information
        i_bottom = np.where(tfr.freqs >= band[0])[0][0]
        i_top = np.where(tfr.freqs <= band[1])[0][-1]
        if dim_freq == 1:
            frequency_data = tfr.data[:, i_bottom:(i_top + 1), :]
        else:
            frequency_data = tfr.data[:, :, i_bottom:(i_top + 1), :]
        mean_band_power = frequency_data.mean(dim_freq, keepdims=True)
        new_data.append(mean_band_power)
        new_freqs.append(band[0])
    # combine band powers along th efrequency axis
    tfr.data = np.concatenate(new_data, dim_freq)
    tfr.freqs = np.asarray(new_freqs)
    return tfr


def log_transform(tfr):
    tfr.data = np.log10(tfr.data)
    return tfr


def z_transform_all(tfr):
    """Z transforms based on mean and sd from all epochs

    Transforms per frequency and electrode. Differs from z_transform_baseline
    which calculates sd in the same way but takes only mean from the baseline
    period per each epoch

    Parameters
    ----------
    tfr : [type]
        [description]
    Returns
    -------
    mne.EpochsTFR
    """
    means, sds = mnestats.epochs_means_sds(tfr)
    tfr.data = (tfr.data - means)/sds
    return tfr


def z_transform_baseline(tfr, baseline):
    """z transforms epoched TFRs based on baseline and sd from all epochs

    the mean is taken from the baseline, but the standard deviation is calculated
    across all epochs and all times. Therefore you shoudl pass here a FULL epochsTFR,
    not just the wanted events (because then the standard deviation could be biased)

    Parameters
    ----------
    tfr : mne.EpochsTFR
        EpochedTFR to be transformed
    baseline : tuple(float, float)
        baseline to calulate mean to be substracted

    Returns
    -------
    mne.EpochsTFR
        Z-transformed object
    """
    tfr = tfr.apply_baseline(baseline, mode="mean")
    _, sds = mnestats.epochs_means_sds(tfr)
    tfr.data = tfr.data/sds
    return tfr


