from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
from functions import helpers
import numpy as np
import pandas as pd


def load_raw(data_path, frequency, montage=None):
    data = eegrd.read_mat(data_path)
    # data = data * 1e-06  # because of stupid scaling
    raw = eegrd.numpy_mne(data, frequency, montage)
    return raw


def load_matlab_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = helpers.remove_unnamed(pd_events)
    return(pd_events)


def load_unity_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = pd_events.drop(['trialIDs', 'pointingError'], 1)
    pd_events = helpers.remove_unnamed(pd_events)
    pd_events = pd.melt(pd_events, id_vars = ['type'])
    pd_events['name'] = pd_events['variable'] + '_' + pd_events['type']
    pd_events = pd_events.drop(['type', 'variable'], 1)
    pd_events = pd_events.rename(index=str, columns = {"value": "time"})
    return pd_events


def pd_to_mne_events(pd_events, frequency):
    event_types = pd_events.name.unique()
    event_nums = list(range(1,  event_types.size + 1))
    mapping =  dict(zip(event_types, event_nums))
    pd_frame = pd_events.replace({'name': mapping})
    pd_frame = pd_frame.sort_values(by = 'time')
    events_second_col = [0] * pd_frame.shape[0]
    events = np.array([pd_frame.time * frequency, events_second_col, pd_frame.name])
    events = events.astype(int)
    events[0, :] = add_one_to_duplicates(events[0, :])
    events = events.transpose()
    return events, mapping


def add_one_to_duplicates(arr):
    dup_ids = helpers.find_duplicates(arr)
    arr[dup_ids] += 1
    if len(helpers.find_duplicates(arr))> 0:
        return add_one_to_duplicates(arr)
    else:
        return arr


def create_mne_events(events_path, frequency):
    pd_events = load_matlab_events(events_path)
    return pd_to_mne_events(pd_events, frequency)


def clear_pd(pd_events):
    pd_events = pd_events[pd_events.time > 0]
    return pd_events