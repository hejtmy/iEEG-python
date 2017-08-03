import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats
from functions import mne_plot_helpers as mneplothelp

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\IntracranialElectrodes\\Data\\p83\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p83\\"

path_montage = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\p83_montage.csv"
path_montage_referenced = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\p83_montage_referenced.csv"
path_original_bva =  base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_250.mat"
path_perhead_bva = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_bip_bva =  base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_events_bva = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\p83_BVA.csv"

FREQUENCY = 250

pd_montage = readeegr.read_montage(path_montage)
pd_montage_referenced = readeegr.read_montage(path_montage_referenced)

# loading BVA data
#raw_original_bva = mneprep.load_raw(path_original_bva, FREQUENCY)
raw_perhead_bva = mneprep.load_raw(path_perhead_bva, FREQUENCY, montage = pd_montage_referenced)

pd_events = mneprep.load_matlab_events(path_events_bva)
mne_events_bva, mapp_bva = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

## PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_bva, pd_montage_referenced, 'Hip')
pick_perhead_hip_names = mne.pick_info(raw_perhead_bva.info, pick_perhead_hip)['ch_names']
pick_perhead_phg = mnehelp.picks_all_localised(raw_perhead_bva, pd_montage_referenced, 'PHG')
pick_perhead_phg_names = mne.pick_info(raw_perhead_bva.info, pick_perhead_phg)['ch_names']
pick_perhead_all = mnehelp.picks_all(raw_perhead_bva)

## Epoching
epochs_perhead_bva = mne.Epochs(raw_perhead_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3, detrend = 1)
bad_epochs_perhead = [14, 15, 28, 41, 42, 45, 49, 50, 57, 58, 67, 87, 89, 90, 91, 95, 98, 99, 104, 109, 110, 113, 123, 124, 125, 126, 127, 128, 129, 131, 135, 141]
epochs_perhead_bva.drop(bad_epochs_perhead)
#epochs_perhead_bva.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)
#print(mnehelp.get_dropped_epoch_indices(epochs_perhead_bva.drop_log))

## TIME FREQUENCY
freqs = np.arange(2, 11, 1)
n_cycles = 6

# ONSETS
power_onset_perhead_bva = tfr_morlet(epochs_perhead_bva['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)
power_stop_perhead_bva = tfr_morlet(epochs_perhead_bva['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False)

power_trials_onset_perhead_bva = tfr_morlet(epochs_perhead_bva['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)
power_trials_stop_perhead_bva = tfr_morlet(epochs_perhead_bva['stops_500_1500'], freqs = freqs, n_cycles = n_cycles, picks = pick_perhead_all, return_itc = False, average = False)

lfo_bands = [[2, 4], [4, 9]]
power_onset_perhead_bva_lfo = mnehelp.band_power(power_onset_perhead_bva, lfo_bands)
power_stop_perhead_bva_lfo = mnehelp.band_power(power_stop_perhead_bva, lfo_bands)

power_trials_onset_perhead_bva_lfo = mnehelp.band_power(power_trials_onset_perhead_bva, lfo_bands)
power_trials_stop_perhead_bva_lfo = mnehelp.band_power(power_trials_stop_perhead_bva, lfo_bands)


## BASELINES
baseline = (-2, -1)
power_stop_perhead_bva_lfo_bas = power_stop_perhead_bva_lfo.copy().apply_baseline(mode='ratio', baseline = (-0.5, 0))
power_onset_perhead_bva_lfo_bas = power_onset_perhead_bva_lfo.copy().apply_baseline(mode='ratio', baseline = (-0.5, 0))

power_trials_stop_perhead_bva_lfo_bas = power_trials_stop_perhead_bva_lfo.copy().apply_baseline(mode='ratio', baseline = (-0.5, 0))
power_trials_onset_perhead_bva_lfo_bas = power_trials_onset_perhead_bva_lfo.copy().apply_baseline(mode='ratio', baseline = (-0.5, 0))

## PLOTS
box = mnehelp.custom_box_layout(pick_perhead_hip_names, 3)
plot_picks_hip = range(0, len(pick_perhead_hip))
power_onset_perhead_bva_lfo_bas.plot_topo(picks =pick_perhead_hip, baseline=(-2, -1), mode='logratio', layout=box)

significant_hip = list(pick_perhead_hip[i] for i in [0,1,2,4])
significant_hip_names = list(pick_perhead_hip_names[i] for i in [0,1,2,4])

significant_phg = list(pick_perhead_phg[i] for i in [0,1,3,4])
significant_phg_names = list(pick_perhead_phg_names[i] for i in [0,1,3,4])


mnehelp.plot_power_time([power_stop_perhead_bva_lfo_bas, power_onset_perhead_bva_lfo_bas], significant_hip, 0, event_names = ['stop', 'onset'], pick_names = significant_hip_names)
mnehelp.plot_power_time([power_stop_perhead_bva_lfo_bas, power_onset_perhead_bva_lfo_bas], significant_phg, 0, event_names = ['stop', 'onset'], pick_names = pick_perhead_phg_names)

mnehelp.plot_power_time_average([power_stop_perhead_bva_lfo_bas, power_onset_perhead_bva_lfo_bas], significant_hip, 0, event_names = ['stop', 'onset'], channel_name= ['Hippocampus'])
mnehelp.plot_power_time_average ([power_stop_perhead_bva_lfo_bas, power_onset_perhead_bva_lfo_bas], significant_phg, 0, event_names = ['stop', 'onset'], channel_name= ['PHG'])

mneplothelp.plot_epochs_power(power_trials_onset_perhead_bva_lfo_bas, 0, significant_hip_names)
mneplothelp.plot_epochs_power(power_trials_onset_perhead_bva_lfo_bas, 0, significant_phg_names)

### STATS
wilcox_stop_onset, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_stop_perhead_bva_lfo_bas, power_trials_onset_perhead_bva_lfo_bas, picks = significant_hip_names)
mnestats.plot_wilcox_box(wilcox_stop_onset, FREQUENCY, pick_names = significant_hip_names, cutout = 0.05)

wilcox_stop_onset_phg, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_stop_perhead_bva_lfo_bas, power_trials_onset_perhead_bva_lfo_bas, picks = significant_phg_names)
mnestats.plot_wilcox_box(wilcox_stop_onset_phg, FREQUENCY, pick_names = significant_phg_names, cutout = 0.05)