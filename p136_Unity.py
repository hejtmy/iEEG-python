import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.stats import permutation_cluster_test
from mne.time_frequency import psd_multitaper

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\IntracranialElectrodes\\Data\\p136\\"
#base_path = "U:\\OneDrive\\FGU\\iEEG\\p136\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"

path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p136_montage.csv"

FREQUENCY = 256

# Loading Unnity data
raw_original_vr = mneprep.load_raw(path_original_vr, FREQUENCY)
raw_perhead_vr = mneprep.load_raw(path_perhead_vr, FREQUENCY)
#raw_bip_vr = mneprep.load_raw(path_bip_vr, FREQUENCY)

pd_unity_events = mneprep.load_unity_events(path_unity_events)
pd_matlab_events = mneprep.load_matlab_events(path_onset_events)
pd_events = pd.concat([pd_unity_events, pd_matlab_events])
pd_events = mneprep.clear_pd(pd_events)

# loading montage
pd_montage = readeegr.read_montage(path_montage)
mne_events_vr, mapp_vr = mneprep.pd_to_mne_events(pd_events, FREQUENCY)

##PLOTS
raw_original_vr.plot(events = mne_events_vr, scalings='auto')
raw_perhead_vr.plot(events = mne_events_vr, scalings='auto')
raw_original_vr.info["bads"] = ['SEEG_47']

## Epoching
epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id=mapp_vr, tmin=-3, tmax=3, add_eeg_ref=False)

# PICKS
pick_orig_hip = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Hi')
pick_orig_hip_names = mne.pick_info(raw_original_vr.info, pick_orig_hip)['ch_names']
pick_orig_ins = mnehelp.picks_all_localised(raw_original_vr, pd_montage, 'Ins')
pick_orig_all = mnehelp.picks_all(raw_original_vr)

# TIME FREQ
freqs = np.arange(2, 30, 1)
n_cycles = 6

box = mnehelp.custom_box_layout(pick_orig_hip_names, 3)
plot_pick_orig_hip = range(len(pick_orig_hip))

#PLOTS
epochs_original_vr['onsets_500_1500', 'stops_500_1500'].plot(picks = pick_orig_hip, block=True, scalings='auto')
epochs_original_vr["pointingEnded_Ego", "pointingEnded_Allo"].plot(block=True, scalings='auto')

power_point_orig_hip_vr_ego = tfr_morlet(epochs_original_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False)
power_point_orig_hip_vr_allo = tfr_morlet(epochs_original_vr['pointingEnded_Allo'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False)

power_onset_orig_hip_vr = tfr_morlet(epochs_original_vr['onsets_500_1500'], freqs=freqs, n_cycles = n_cycles,
                                    picks = pick_orig_hip, return_itc=False)
power_stop_orig_hip_vr = tfr_morlet(epochs_original_vr['stops_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False)

power_point_orig_hip_vr_ego.plot_topo(picks=plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)
power_point_orig_hip_vr_allo.plot_topo(picks=plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)

power_onset_orig_hip_vr.plot_topo(picks=plot_pick_orig_hip, baseline=(-3., -2), mode='logratio', layout=box)
power_stop_orig_hip_vr.plot_topo(picks=plot_pick_orig_hip, baseline=(-3, -2), mode='logratio', layout=box)

power_trials_point_orig_hip_ego = tfr_morlet(epochs_original_vr['pointingEnded_Ego'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False, average = False)
power_trials_point_orig_hip_allo = tfr_morlet(epochs_original_vr['pointingEnded_Allo'], freqs=freqs, n_cycles=n_cycles,
                                    picks = pick_orig_hip, return_itc=False, average = False)

power_trials_point_orig_hip_ego.apply_baseline(mode='ratio', baseline=(None, 0))
power_trials_point_orig_hip_allo.apply_baseline(mode='ratio', baseline=(None, 0))

## PERMUTATIONS TESTS
epochs_power_1 = power_trials_point_orig_hip_ego.data[:, 0, :, :]  # only 1 channel as 3D matrix
epochs_power_2 = power_trials_point_orig_hip_allo.data[:, 0, :, :]  # only 1 channel as 3D matrix

threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([epochs_power_1, epochs_power_2],
                             n_permutations=100, threshold=threshold, tail=0)

times = 1e3 * epochs_original_vr['pointingEnded_Ego'].times  # change unit to ms

plt.figure()
# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

plt.imshow(T_obs,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', cmap='gray')
plt.imshow(T_obs_plot,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power ()')

plt.show()

# POWER ESTIMATES
raw_original_vr.plot_psd(picks=pick_orig_hip)

mnehelp.plot_theta_epochs(epochs_original_vr['onsets_500_1500'], 
                          [pick_orig_hip, pick_orig_ins], ['Hippocampus', 'Insula'])

mnehelp.plot_theta_epochs(epochs_original_vr['onsets_500_1500'], 
                          [pick_orig_hip, pick_orig_all], ['Hippocampus', 'All'])

mnehelp.plot_theta_epochs(epochs_original_vr['stops_500_1500'], 
                          [pick_orig_hip, pick_orig_ins], ['Hippocampus', 'Insula'])
