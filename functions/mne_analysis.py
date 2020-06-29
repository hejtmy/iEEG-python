import numpy as np
from mne.time_frequency import tfr_morlet
import mne


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
    tfr.data = np.log(tfr.data)


def z_transform(tfr, per="channel"):
    return None
