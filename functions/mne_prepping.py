from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
from functions import helpers
import numpy as np
import pandas as pd


def load_raw(data_path, frequency, montage=None):
    """Loads eeg data from given filepath and converts to mne raw
    Parameters
    ----------
    data_path : str
        file path to the eeg mat file
    frequency : int
        frequnecy of the recording
    montage : pandas.DataFrame
        montage as loaded by read_montage

    Returns
    -------
    mne.raw eeg
    """
    data = eegrd.read_mat(data_path)
    raw = eegrd.eeg_mat_to_mne(data, frequency, montage)
    return raw


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
