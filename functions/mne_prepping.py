from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd
from functions import helpers
import numpy as np

def load_raw(data_path, frequency):
    data = eegrd.read_mat(data_path)
    data = data * 1e-09  # because of stupid scaling
    raw = eegrd.numpy_mne(data, frequency)
    return raw

def load_matlab_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = helpers.remove_unnamed(pd_events)
    return(pd_events)

def load_unity_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = pd_events.drop(['trialIDs'], 1)
    pd_events = helpers.remove_unnamed(pd_events)
    pd_events = pd_events.melt(id_vars = ['type'])
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
    events = events.transpose()
    return events, mapping

def create_mne_events(events_path, frequency):
    pd_events = load_matlab_events(events_path)
    return pd_to_mne_events(pd_events, frequency)

def clear_pd(pd_events):
    pd_events = pd_events[pd_events.time > 0]
    return pd_events

# What this does is basically every evnet occuring at the same time is shifted one sample to the future - so that we don't get duplicate values in the epochs
def solve_duplicates(pd_events, frequency):
    #genius answer
    # https://stackoverflow.com/questions/35484250/python-counting-cumulative-occurrences-of-values-in-a-pandas-series
    pd_events['add_this'] = (pd_events.groupby('time').cumcount()/frequency)
    pd_events.time = pd_events.time + pd_events.add_this
    pd_events = pd_events.drop("add_this", 1)
    return pd_events