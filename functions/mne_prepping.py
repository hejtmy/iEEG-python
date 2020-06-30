import numpy as np
import pandas as pd
import mne

from functions import read_experiment_data as exprd
from functions import paths
from functions import helpers


def load_preprocessed_events(file_paths):
    """[summary]

    Parameters
    ----------
    file_paths : dictionary
        Dictionary of file paths obtained from functions.paths.prep_files

    Returns
    -----------
    events, mapping : pandas.DataFrame
        pandas.DataFrame dataframe with events and timesincestart
    """
    pd_unity_events = load_unity_events(
            file_paths['experiment']['events_timesinceeegstart'])
    pd_matlab_events = load_matlab_events(file_paths['experiment']['onsets'])
    pd_events = pd.concat([pd_unity_events, pd_matlab_events])
    return pd_events


def load_matlab_events(events_path):
    """Loads events exported from matlab (usually onsets) and formats the table

    Parameters
    ----------
    events_path : str
        fulll file path to the event file

    Returns
    ---------
    formted pandas.DataFrame table with 'event;time' columns. Time designates
    time since the eegstart
    """
    pd_events = exprd.read_events(events_path)
    pd_events = helpers.remove_unnamed(pd_events)
    pd_events = pd_events.drop(columns=['unitytime', 'eegtime'])
    pd_events = pd_events.rename(columns={'timesinceeegstart': 'time'})
    return(pd_events)


def load_unity_events(events_path):
    """Loads unity table and prepares it to structured format

    Parameters
    ----------
    events_path : str
        file path to the event file

    Returns
    --------
    pandas.DataFrame with 'event;time' columns.
    """
    pd_events = exprd.read_events(events_path)
    pd_events = pd_events.drop(['trialId', 'pointingError'], 1)
    pd_events = helpers.remove_unnamed(pd_events)
    pd_events = pd.melt(pd_events, id_vars=['type'])
    pd_events['name'] = pd_events['variable'] + '_' + pd_events['type']
    pd_events = pd_events.drop(['type', 'variable'], 1)
    pd_events = pd_events.rename(index=str, columns={"value": "time"})
    return pd_events


def pd_to_mne_events(pd_events, frequency):
    """Converts pandas.DataFrame to a mne valid evnets to use in Epoching

    Parameters
    ----------
    pd_events : [type]
        [description]
    frequency : [type]
        [description]

    Returns
    -------
    events, mapping : touple
        array with mne valid events []
        mapped dictionary to use in pd_to_mne_events
    """
    pd_events = clear_pd(pd_events)
    event_types = pd_events.name.unique()
    event_nums = list(range(1,  event_types.size + 1))
    mapping = dict(zip(event_types, event_nums))
    pd_frame = pd_events.replace({'name': mapping})
    pd_frame = pd_frame.sort_values(by='time')
    events_second_col = [0] * pd_frame.shape[0]
    events = np.array([pd_frame.time * frequency,
                       events_second_col, pd_frame.name])
    events = events.astype(int)
    events[0, :] = add_one_to_duplicates(events[0, :])
    events = events.transpose()
    return events, mapping


def add_one_to_duplicates(arr):
    """
    """
    dup_ids = helpers.find_duplicates(arr)
    arr[dup_ids] += 1
    if len(helpers.find_duplicates(arr)) > 0:
        return add_one_to_duplicates(arr)
    else:
        return arr


def clear_pd(pd_events):
    """Removes faulty events
    """
    pd_events = pd_events[pd_events.time > 0]
    return pd_events


def create_montage(eeg, pd_montage):
    """[summary]

    Doesn't work. Started working on it, but couldn't plot due to renderer
    https://mne.tools/stable/auto_tutorials/misc/plot_ecog.html#sphx-glr-auto-tutorials-misc-plot-ecog-py
    Parameters
    ----------
    eeg : [type]
        [description]
    pd_montage : [type]
        [description]
    """
    coords = pd_montage[['MNI_x', 'MNI_y', 'MNI_z']]/1000
    tuples = [tuple(x) for x in coords.to_numpy()]
    mne_dig_montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(eeg.info['ch_names'], tuples)),
        coord_frame='head')
    info = eeg.info.set_montage(mne_dig_montage)
    return(info)


def write_bad_epochs(epochs, file_paths, append=''):
    """[summary]

    Parameters
    ----------
    epochs : mne.Epochs
        [description]
    file_paths : [type]
        [description]
    append : str, optional
        [description], by default ''
    """
    bad_epochs = np.where(epochs.drop_log)
    filepath = paths.bad_epochs_path(file_paths, append)
    np.savetxt(filepath, bad_epochs, fmt='%1.0i', delimiter=',')


def read_bad_epochs(file_paths, append=''):
    filepath = paths.bad_epochs_path(file_paths, append)
    return np.genfromtxt(filepath, dtype='int64', delimiter=',')


def save_tfr_epochs(epochs, file_paths, append='', overwrite=False):
    """Saves convolved tfr to a predefined file

    Parameters
    ----------
    file_paths : dictionary of filepaths
        [description]
    epochs : mne.EpochsTFR or mne.AverageTFR
        [description]
    append : str, optional
        appendix to a name to separate various convolutions, by default ''
    overwrite : bool, optional

    """
    filepath = paths.tfr_epochs_path(file_paths, append)
    epochs.save(filepath, overwrite=overwrite)


def load_tfr_epochs(file_paths, append=''):
    """[summary]

    Parameters
    ----------
    file_paths : dictionary of str
        list of paths as generated by paths preppring functions 
    append : str, optional
        appendix to a name to separate various convolutions, by default ''
    """
    filepath = paths.tfr_epochs_path(file_paths, append)
    return mne.time_frequency.read_tfrs(filepath)[0]
