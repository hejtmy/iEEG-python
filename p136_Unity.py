import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

################### PREPARATION ----------------------------------

base_path = "D:\\IntracranialElectrodes\\Data\\p136\\"
#base_path = "U:\\OneDrive\\FGU\\iEEG\\p136\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"

path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_montage.csv"

FREQUENCY = 256

# loading montage
pd_montage = readeegr.read_montage(path_montage) 

# Loading Unity data
raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY, pd_montage)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
#raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

##PLOTS
#raw_original_vr.plot(events = mne_events_vr, scalings='auto')
#raw_perhead_vr.plot(events = mne_events_vr, scalings='auto')
raw_original_vr.info["bads"] = ['SEEG_47']

## Epoching
epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)
epochs_perhead_vr = mne.Epochs(raw_perhead_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3, add_eeg_ref = False)

# PICKS
pick_orig_hip = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Hi')
pick_orig_hip_names = mne.pick_info(raw_original_vr.info, pick_orig_hip)['ch_names']
pick_orig_ins = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Ins')
pick_orig_all = mnehelp.picks_all(raw_original_vr)

# BAD EPOCHS
bad_epochs = [10, 42, 51, 52, 74, 82, 92, 93, 114, 129, 130, 131, 161, 165, 181, 185, 222, 255, 268, 269, 275, 291, 299,
              300, 301, 302, 311, 323, 324, 325, 326, 327, 328, 346, 347, 348, 349, 353, 366, 367, 368, 369, 370, 376,
              397, 398, 427, 435, 438, 450, 461, 489, 490, 491, 492, 493, 494, 499, 518]
epochs_original_vr.drop(bad_epochs)
epochs_perhead_vr.drop(bad_epochs)
#epochs_original_vr.plot(block = True, scalings = 'auto', picks=pick_orig_hip)

epochs_onsets_stops = epochs_original_vr['onsets_500_1500', 'stops_500_1500']
# can be obtained with mnehelp.get_dropped_epoch_indices(epochs_onsets_stops.drop_log)
# bad_onset_epochs = [41, 42, 57, 63, 64, 69, 70, 82, 108, 144, 165, 172, 173, 186, 201, 204, 215, 226, 227, 228]
# epochs_onsets_stops.drop(bad_onset_epochs)
# epochs_onsets_stops.plot(picks = pick_orig_hip, block = True, scalings = 'auto')

epochs_pointing_allo_ego = epochs_original_vr["pointingEnded_Ego", "pointingEnded_Allo"]
#epochs_pointing_allo_ego.plot(picks = pick_orig_hip, block = True, scalings = 'auto')
# can be obtained with mnehelp.get_dropped_epoch_indices(epochs_pointing_allo_ego.drop_log)
# bad_point_epochs = [1, 3, 22, 24, 30, 31, 33]
# epochs_pointing_allo_ego.drop(bad_point_epochs)

########### ANALYSES ----------------------------------

# TIME FREQ
freqs = np.arange(2, 11, 1)
n_cycles = 6
box = mnehelp.custom_box_layout(pick_orig_hip_names, 3)
plot_pick_orig_hip = range(len(pick_orig_hip))

power_point_orig_hip_vr_ego = tfr_morlet(epochs_original_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False)
power_point_orig_hip_vr_allo = tfr_morlet(epochs_original_vr['pointingEnded_Allo'], freqs=freqs, n_cycles=n_cycles,picks = pick_orig_hip, return_itc=False)

power_onset_orig_hip_vr = tfr_morlet(epochs_original_vr['onsets_500_1500'], freqs = freqs, n_cycles = n_cycles,
                                    picks = pick_orig_hip, return_itc = False)
power_stop_orig_hip_vr = tfr_morlet(epochs_original_vr['stops_500_1500'], freqs = freqs, n_cycles = n_cycles,
                                    picks = pick_orig_hip, return_itc = False)

#event X electrode X freqs X time
power_trials_point_orig_hip_ego = tfr_morlet(epochs_original_vr['pointingEnded_Ego'], freqs=freqs, n_cycles = n_cycles,picks = pick_orig_hip, return_itc = False, average = False)
power_trials_point_orig_hip_allo = tfr_morlet(epochs_original_vr['pointingEnded_Allo'], freqs = freqs, n_cycles = n_cycles,picks = pick_orig_hip, return_itc = False, average = False)

### LFO BANDS
lfo_bands = [[2, 4], [4, 9]]
power_point_orig_hip_vr_ego_lfo = mnehelp.band_power(power_point_orig_hip_vr_ego, lfo_bands)
power_point_orig_hip_vr_allo_lfo = mnehelp.band_power(power_point_orig_hip_vr_allo, lfo_bands)

power_trial_point_orig_hip_vr_ego_lfo = mnehelp.band_power(power_trials_point_orig_hip_ego, lfo_bands)
power_trial_point_orig_hip_vr_allo_lfo = mnehelp.band_power(power_trials_point_orig_hip_allo, lfo_bands)

## PLOTS
power_point_orig_hip_vr_ego.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)
power_point_orig_hip_vr_allo.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)

power_point_orig_hip_vr_ego_lfo.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)
power_point_orig_hip_vr_allo_lfo.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)


power_onset_orig_hip_vr.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)
power_stop_orig_hip_vr.plot_topo(picks = plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)


power_trials_point_orig_hip_ego.apply_baseline(mode='ratio', baseline=(None, 0))
power_trials_point_orig_hip_allo.apply_baseline(mode='ratio', baseline=(None, 0))

# POWER ESTIMATES
raw_original_vr.plot_psd(picks = pick_orig_hip)

mnehelp.plot_psd_epochs([epochs_original_vr['onsets_500_1500']], [pick_orig_hip, pick_orig_ins], 1, 8, 0, 1.5, ['Hippocampus', 'Insula'], ['Onsets'])

mnehelp.plot_psd_epochs([epochs_original_vr['onsets_500_1500']], [pick_orig_hip, pick_orig_all], 1, 8, ['Hippocampus', 'All'], ['Onsets'])

mnehelp.plot_psd_epochs([epochs_original_vr['stops_500_1500'], epochs_original_vr['onsets_500_1500']], [pick_orig_hip], 1, 16,  ['Hippocampus'], ['stops', 'onsets'])

mnehelp.plot_psd_epochs_separate([epochs_original_vr['stops_500_1500'], epochs_original_vr['onsets_500_1500']], [pick_orig_hip], 1, 16, ['Hippocampus'], ['stops', 'onsets'])
mnehelp.plot_psd_epochs_separate([epochs_original_vr], [pick_orig_hip], 1, 16, ['Hippocampus'])


for pick in list(pick_orig_hip):
    mnehelp.plot_psd_epochs([epochs_original_vr['pointingEnded_Ego'], epochs_original_vr['pointingEnded_Allo']], [[pick]],  1, 8,  -1.5, 0, ['Hippocampus' + str(pick)], ['pointingEnded_Ego', 'pointingEnded_Allo'])


for pick in list(pick_orig_hip):
    mnehelp.plot_psd_epochs([epochs_original_vr['stops_500_1500'], epochs_original_vr['onsets_500_1500']], [[pick]],  1, 8,  0, 1.5, ['Hippocampus' + str(pick)], ['stops_500_1500', 'onsets_500_1500'])

wilcox_allo_ego, wilcox_freqs = mnestats.wilcox_tfr_power(power_trials_point_orig_hip_ego, power_trials_point_orig_hip_allo)

wilcox_allo_ego_lfo, wilcox_freqs_lfo = mnestats.wilcox_tfr_power(power_trial_point_orig_hip_vr_ego_lfo, power_trial_point_orig_hip_vr_allo_lfo)

mnestats.plot_wilcox_box(wilcox_allo_ego, 256, freqs = wilcox_freqs)
mnestats.plot_wilcox_box(wilcox_allo_ego_lfo, 256, freqs = wilcox_freqs_lfo)

mnestats.plot_wilcox(wilcox_allo_ego, 0, 256)
for channel in range(len(pick_orig_hip)):
    mnestats.plot_wilcox(wilcox_allo_ego, channel, 256, freqs = wilcox_freqs)