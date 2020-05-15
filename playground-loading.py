import mne
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
# from functions import mne_stats as mnestats
from functions import paths

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
frequency = paths.get_frequency(paths.eeg_path(paths.unity_alloego_path(base_path, participant)))

# loading montage
pd_montage = readeegr.read_montage(file_paths['montage']['original']) 
pd_montage_referenced = readeegr.read_montage(file_paths['montage']['referenced']) 

# Loading Unity data
raw_original = mneprep.load_raw(file_paths['EEG']['base'], frequency, pd_montage)
raw_perhead = mneprep.load_raw(file_paths['EEG']['perHeadbox'], frequency, pd_montage_referenced)
raw_perelectrode = mneprep.load_raw(file_paths['EEG']['perElectrode'], frequency, pd_montage_referenced)
raw_bipolar = mneprep.load_raw(file_paths['EEG']['bipolar'], frequency, pd_montage_referenced)

pd_unity_events = mneprep.load_unity_events(file_paths['experiment']['events_timesinceeegstart'])
pd_matlab_events = mneprep.load_matlab_events(file_paths['experiment']['onsets'])
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)

mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, frequency)

# Epoching
epochs_original = mne.Epochs(raw_original, mne_events, event_id=events_mapp,
                             tmin=-3, tmax=3)
epochs_original.plot(scalings=scalings)

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead)

# Playing
raw_original.plot(scalings=scalings)
raw_original.plot_psd(fmax=100, picks=['seeg'], average=False)

raw_perhead.plot()
raw_bipolar.plot()