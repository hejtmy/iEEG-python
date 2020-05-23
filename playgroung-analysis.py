import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import paths

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
frequency = readeegr.get_frequency(paths.eeg_path(paths.unity_alloego_path(base_path, participant)))

# loading montage
pd_montage = readeegr.read_montage(file_paths['montage']['original']) 
pd_montage_referenced = readeegr.read_montage(file_paths['montage']['referenced']) 

# Loading Unity data
raw_original = mneprep.load_raw(file_paths['EEG']['base'], frequency, pd_montage)
raw_perhead = mneprep.load_raw(file_paths['EEG']['perHeadbox'], frequency, pd_montage_referenced)
raw_perelectrode = mneprep.load_raw(file_paths['EEG']['perElectrode'], frequency, pd_montage_referenced)
raw_bipolar = mneprep.load_raw(file_paths['EEG']['bipolar'], frequency, pd_montage_referenced)

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, frequency)

raw_eeg = raw_original

# Epoching
epochs = mne.Epochs(raw_eeg, mne_events, event_id=events_mapp,
                    tmin=-3, tmax=3)
epochs.plot(scalings=scalings)

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead)

# Playing
raw_eeg.plot(scalings=scalings)
raw_eeg.plot_psd(fmax=100, picks=['seeg'], average=False, show=False)

# TIME FREQ
freqs = np.arange(1, 11, 1)
n_cycles = 6
box = mnehelp.custom_box_layout(pick_perhead_hip_names, 3)
plot_pick_perhead_hip = range(len(pick_perhead_hip))