import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import mne_loading as loader
from functions import paths
from functions import mne_analysis as mneanalysis

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
eeg, montage = loader.load_eeg(file_paths, 'bipolar')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Preprocessing
eeg.notch_filter(50)

# Epoching
epochs = mne.Epochs(eeg, mne_events, event_id=events_mapp, tmin=-3, tmax=3)
epochs['onsets_500_1500'].plot(scalings=scalings, n_epochs=6, n_channels=15)

# PICKS
pick_hip = mnehelp.picks_all_localised(eeg, montage, 'Hi')
pick_hip_names = mne.pick_info(eeg.info, pick_hip)['ch_names']
pick_all = mnehelp.picks_all(eeg, montage)
pick_all_names = mne.pick_info(eeg.info, pick_all)['ch_names']

# Playing
eeg.plot(scalings=scalings)
eeg.plot_psd(fmax=100, picks=pick_hip, average=False)

# TIME FREQ
freqs = np.arange(2, 12, 0.5)
n_cycles = 6

events = ['onsets_500_1500', 'stops_500_1500', 
          'pointingStarted_Ego', 'pointingEnded_Ego',
          'pointingStarted_Allo', 'pointingEnded_Allo']
morlet = mneanalysis.morlet_all_events(
    epochs, freqs, n_cycles, events=['onsets_500_1500'])

box = mnehelp.custom_box_layout(pick_all_names, 4)
box = mne.channels.layout.make_grid_layout(eeg.info, pick=pick_all)
morlet['onsets_500_1500'].average().plot_topo(
    picks=pick_all, layout=box, mode='logratio', baseline=(-1, -0.5))
morlet['onsets_500_1500'].average().plot(110, baseline=(-1, -0.5), mode='zlogratio')

baseline = (-1, -0.5)
mode = 'ratio'
lfo_bands = [[2, 4], [4, 9]]

morlet = mneanalysis.convolutions_apply_baselines(morlet, baseline, mode)
morlet = mneanalysis.convolutions_band_power(morlet, lfo_bands)
