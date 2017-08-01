import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power


base_path = "D:\\IntracranialElectrodes\\Data\\p129\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p129\\"

path_original_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_bip_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"
path_perelectrode_vr = base_path + "\\UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_montage.csv"
path_montage_referenced = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p129_montage_referenced.csv"

FREQUENCY = 256
runfile('M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python/base_setup.py', wdir='M:/Vyzkum/AV/FGU/IntracranialElectrodes/iEEG-python')

# PICKS
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_vr.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_vr, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead_vr)

# BAD EPOCHS
bad_epochs_perhead = [19, 51, 52, 185, 186, 204, 205, 287, 288, 289, 302, 367, 368, 373, 374, 375, 376, 377, 378, 418, 419, 488, 489, 490, 504, 505, 506, 507]
epochs_perhead_vr.drop(bad_epochs_perhead)
#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)

iter_freqs = [
    ('Delta', 1, 4),
    ('Theta', 5, 8)
]

events_to_do = {'onsets_500_1500': 14,
 'pointingEnded_Allo': 9,
 'pointingEnded_Ego': 10,
 'pointingStarted_Allo': 7,
 'pointingStarted_Ego': 8,
 'stops_500_1500': 15
 }

# set epoching parameters
event_id, tmin, tmax = 1, -3., 3.
baseline = (-3, -2)
frequency_map = list()
    
import matplotlib.pyplot as plt 
def get_gfp_ci(average, n_bootstraps=2000):
    """get confidence intervals from non-parametric bootstrap"""
    indices = np.arange(len(average.ch_names), dtype=int)
    gfps_bs = np.empty((n_bootstraps, len(average.times)))
    for iteration in range(n_bootstraps):
        bs_indices = rng.choice(indices, replace=True, size=len(indices))
        gfps_bs[iteration] = np.sum(average.data[bs_indices] ** 2, 0)
    gfps_bs = mne.baseline.rescale(gfps_bs, average.times, baseline=(None, 0))
    ci_low, ci_up = np.percentile(gfps_bs, (2.5, 97.5), axis=0)
    return ci_low, ci_up

rng = np.random.RandomState(42)

def plot_event_gfo(raw_perhead_vr, events_to_do, mne_events_vr, iter_freqs, picks):
    frequency_map = list()
    for band, fmin, fmax in iter_freqs:
    # (re)load the data to save memory
        raw = raw_perhead_vr.copy()
        raw = raw.pick_channels(picks)
        # bandpass filter and compute Hilbert
        raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   fir_design='firwin')
        raw.apply_hilbert(n_jobs=1, envelope=False)
        epochs = mne.Epochs(raw, mne_events_vr, event_id = events_to_do, tmin = -3, tmax = 3, preload=True)
        # remove evoked response and get analytic signal (envelope)
        epochs.subtract_evoked()  # for this we need to construct new epochs.
        epochs = mne.EpochsArray(data=np.abs(epochs.get_data()), info = epochs.info, tmin=epochs.tmin)  
        frequency_map.append(((band, fmin, fmax), epochs.average()))
        
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    colors = plt.cm.viridis((0.1, 0.35, 0.75, 0.95))    
    for ((freq_name, fmin, fmax), average), color, ax in zip(frequency_map, colors, axes.ravel()[::-1]):
        times = average.times * 1e3
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
        ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
        ax.plot(times, np.zeros_like(times), linestyle='--', color='red', linewidth=1)
        ci_low, ci_up = get_gfp_ci(average)
        ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
        ax.grid(True)
        ax.set_ylabel('GFP')
        ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.set_xlim(-3050, 3050)
    axes.ravel()[-1].set_xlabel('Time [ms]')

plot_event_gfo(raw_perhead_vr, {'onsets_500_1500': 14}, mne_events_vr, iter_freqs, pick_perhead_hip_names)
plot_event_gfo(raw_perhead_vr, {'stops_500_1500': 15}, mne_events_vr, iter_freqs, pick_perhead_hip_names)
