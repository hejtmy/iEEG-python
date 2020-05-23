from mne.time_frequency import tfr_morlet


def morlet_all_events(epochs, freqs, n_cycles,
                           events=[], return_itc=False):
    """runs mne.time_frequency.tfr_morlet on all evens in the epoch
    object and returns a dictionary with all computed

    Parameters
    ----------
    epochs : mne.Epoch
        object with the epochs
    freqs : [type]
        what frequencies to convolve.See tfr_morlet for details
    n_cycles : [type]
        number of cycles. See tfr_morlet for details
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
        convolutions[event] = tfr_morlet(
            epochs[event], freqs=freqs, n_cycles=n_cycles,
            return_itc=return_itc)
    return convolutions