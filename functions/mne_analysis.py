import numpy as np
from mne.time_frequency import tfr_morlet


def morlet_all_events(epochs, freqs, n_cycles, average=True,
                      events=[], return_itc=False):
    """runs mne.time_frequency.tfr_morlet on all evens in the epoch
    object and returns a dictionary with all computed

    Parameters
    ----------
    epochs : mne.Epoch
        object with the epochs
    freqs : touple of int
        what frequencies to convolve.See tfr_morlet for details
    n_cycles : int
        number of cycles. See tfr_morlet for details
    average : bool
        shoudl it return AverageTFR or EpochsTFR
    events : list, optional
        list of event names. If empty, calculates convolusions
        for all events. by default []
    return_itc : bool, optional
        see tfr_morlet for details. by default False

    Returns
    -------
    dictionary
        returns dictionary of convolutions
    """
    all_events = epochs.event_id.keys()
    if len(events) == 0:
        events = all_events
    convolutions = dict()
    for event in events:
        if event not in all_events:
            continue
        print(f'Convolving {event}')
        convolutions[event] = tfr_morlet(
            epochs[event], freqs=freqs, n_cycles=n_cycles,
            return_itc=return_itc, average=average)
    return convolutions


def convolutions_apply_baselines(convolutions, baseline, mode, events=[]):
    """Apply baseline to all convolutions. convolutions need to be in
    AverageTFR or EpochsTFR format which has apply_baseline method

    Parameters
    ----------
    convolutions : dictionary of AverageTFR
        Dictionary of all events
    baseline : touple of numbers
        seconds of the event to baseline
    mode : str
        type of bsaeline. See apply_baseline for more information
    events : list, optional
        list of named events from convolutions to baseline.
        If empty, baselines all events. By default []

    Returns
    -------
    dict
        dictionary of baselined convolutions
    """
    all_events = convolutions.keys()
    if len(events) == 0:
        events = all_events
    for event in events:
        if event not in all_events:
            continue
        convolutions[event] = convolutions[event].apply_baseline(
            baseline=baseline, mode=mode)
    return convolutions


def convolutions_band_power(convolutions, frequency_bands):
    for event in convolutions.keys():
        convolutions[event] = band_power(convolutions[event], frequency_bands)
    return convolutions


def band_power(tfr, bands):
    """Averages TFR object into given bands

    Parameters
    ----------
    tfr : mne.AverageTFR
        object ot average into
    bands : list of num touples
        list of  of bottom inclusive touples of wanted bands.
        e.g. [(2, 4),(4, 9)] will select bands [2, 3] and [4, 8]
    Returns
    -------
    Returns the same object as was passed but with recalculated powers
        [description]

    Example
    -------
    """
    if len(tfr.data.shape) not in [3, 4]:
        print("This doesn¨t look like a TFR file. Dimensions are not correct")
        return None
    is_averaged = len(tfr.data.shape) == 3
    # at what index is the freq dimension/axis
    # EpochsTFR are (n_epochs, n_channels, n_freqs, n_times)
    # AverageTFR are (n_channels, n_freqs, n_times)
    if is_averaged:
        dim_freq = 1
    else:
        dim_freq = 2

    new_data = []
    new_freqs = []
    for band in bands:
        # finds indices of these freqs
        i_bottom = np.where(tfr.freqs >= band[0])[0][0]
        i_top = np.where(tfr.freqs < band[1])[0][-1]
        # tfr data are channel, freq, time - selecting all freqs of given band
        if is_averaged:
            frequency_data = tfr.data[:, i_bottom:(i_top + 1), :]
        else:
            frequency_data = tfr.data[:, :, i_bottom:(i_top + 1), :]
        mean_band_power = frequency_data.mean(dim_freq, keepdims=True)
        new_data.append(mean_band_power)
        new_freqs.append(band[0])
    band_tfr = tfr.copy()
    # combiine band powers along th efrequency axis
    band_tfr.data = np.concatenate(new_data, dim_freq)
    band_tfr.freqs = np.asarray(new_freqs)
    return band_tfr
