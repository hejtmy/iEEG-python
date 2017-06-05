from functions import read_eeg as eegrd
from functions import read_experiment_data as exprd

def load_raw(data_path, frequency):
    data = eegrd.read_mat(data_path)
    data = data * 1e-09  # because of stupid scaling
    raw = eegrd.numpy_mne(data, frequency)
    return raw


def load_matlab_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = pd_events.drop(['Unnamed: 2'], 1)
    return(pd_events)

def load_unity_events(events_path):
    pd_events = exprd.read_events(events_path)
    pd_events = pd_events.drop(['trialIDs','Unnamed: 8'], 1)
    pd_events = pd_events.melt(id_vars = ['type'])
    pd_events['name'] = pd_events['variable'] + '_' + pd_events['type']
    pd_events = pd_events.drop(['type', 'variable'], 1)
    pd_events = pd_events.rename(index=str, columns={"value": "time"})
    return pd_events

def pd_to_mne_events(pd_events, frequency):
    mne_events, mapp = exprd.mne_epochs_from_pd(pd_events, frequency)
    return mne_events, mapp

def create_mne_events(events_path, frequency):
    pd_events = load_matlab_events(events_path)
    return pd_to_mne_events(pd_events, frequency)