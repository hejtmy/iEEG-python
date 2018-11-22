import mne
import numpy as np
from functions import mne_helpers as mnehelp

from functions import mne_stats as mnestats
from functions import mne_plot_helpers as mneplot

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

################### PREPARATION ----------------------------------

base_path = "D:\\IntracranialElectrodes\\Data\\p187\\"
base_path = "U:\\OneDrive\\FGU\\iEEG\\p187\\"

path_original_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_256.mat"
path_perhead_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_256.mat"
path_perelectrode_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_perElectrode_256.mat"
path_bip_vr = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\prep_bipolar_256.mat"

path_unity_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p187_unity.csv"
path_onset_events = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p187_onsets.csv"
path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p187_montage.csv"
path_montage_referenced = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p187_montage_referenced.csv"

FREQUENCY = 256
exec(open("base_setup.py").read(), globals())

raw_perhead_vr.plot(events = mne_events_vr, scalings = 'auto', event_id=mapp_vr) #coloring events 4 and 5
raw_perhead_vr.plot(events = mne_events_vr, scalings = 'auto', event_id=mapp_vr) #coloring events 4 and 5

epochs_original_vr = mne.Epochs(raw_original_vr, mne_events_vr, event_id = mapp_vr, tmin = -3, tmax = 3)
epochs_original_vr['onsets_500_1500'].plot(scalings='auto')

# Hipocampus CA 3 are 61 and 62

epochs_perhead_vr['onsets_500_1500'].plot_image(61, cmap='interactive', sigma=1., scalings='auto')
epochs_perhead_vr['onsets_500_1500'].plot_psd(fmin=2., fmax=40.)

# Morlet
freqs = np.logspace(*np.log10([2, 20]), num=4)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs_original_vr['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

power.plot([61], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[61])
power.plot([62], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[62])