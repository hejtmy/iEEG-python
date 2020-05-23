import mne

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import mne_loading as loader
from functions import paths

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
eeg, montage = loader.load_eeg(file_paths, 'original')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Epoching
epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp,
                    tmin=-3, tmax=3)
epochs.plot(scalings=scalings)

# PICKS
pick_hip = mnehelp.picks_all_localised(eeg, montage, 'Hi')
pick_perhead_hip_names = mne.pick_info(eeg.info, pick_hip)['ch_names']
pick_perhead_all = mnehelp.picks_all(eeg, montage)

# Playing
eeg.plot(scalings=scalings)
eeg.plot_psd(fmax=100, picks=pick_hip, average=False)
