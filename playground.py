import mne
# import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
# from functions import mne_stats as mnestats
from functions import paths

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p129'
file_paths = paths.prep_unity_alloego_files(base_path, participant)

frequency = paths.get_frequency(paths.eeg_path(paths.unity_alloego_path(base_path, participant)))

# loading montage
pd_montage = readeegr.read_montage(file_paths['montage']['original']) 
pd_montage_referenced = readeegr.read_montage(file_paths['montage']['referenced']) 

# Loading Unity data
raw_original_vr = mneprep.load_raw(file_paths['EEG']['base'], frequency, pd_montage)
raw_perhead_vr = mneprep.load_raw(file_paths['EEG']['perHeadbox'], frequency, pd_montage_referenced)
raw_perelectrode_vr = mneprep.load_raw(file_paths['EEG']['perElectrode'], frequency, pd_montage_referenced)
raw_bipolar_vr = mneprep.load_raw(file_paths['EEG']['bipolar'], frequency, pd_montage_referenced)
raw_bip_vr = mneprep.load_raw(file_paths['EEG']['bipolar'], frequency, pd_montage_referenced)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, frequency)

## Epoching
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3)

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)
