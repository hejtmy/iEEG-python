import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\IntracranialElectrodes\\Data\\p83\\"
#base_path = "U:\\OneDrive\\FGU\\iEEG\\p83\\"

path_montage = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\p83_montage.csv"

path_original_bva =  base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_250.mat"
path_perhead_bva = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_perHeadbox_250.mat"
path_bip_bva =  base_path + "BVAAlloEgo\\EEG\\Preprocessed\\prep_bipolar_250.mat"
path_events_bva = base_path + "BVAAlloEgo\\EEG\\Preprocessed\\p83_BVA.csv"

FREQUENCY = 250

pd_montage = readeegr.read_montage(path_montage)

# loading BVA data
raw_original_bva = mneprep.load_raw(path_original_bva, FREQUENCY)
#raw_perhead_bva = mneprep.load_raw(path_perhead_bva, FREQUENCY)
raw_bip_bva = mneprep.load_raw(path_bip_bva, FREQUENCY)
pd_events = mneprep.load_matlab_events(path_events_bva)
mne_events_bva, mapp_bva = mneprep.pd_to_mne_events(pd_events, FREQUENCY)


# {'c': 1, 'f': 2, 'g': 3, 'onsets_500_1500': 4, 'stops_500_1500': 5}
raw_original_bva.plot(events=mne_events_bva, scalings='auto',
                     event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})
raw_bip_bva.plot(events=mne_events_bva, scalings='auto',
                 event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})

raw_original_bva.info["bads"] = ['SEEG_55', 'SEEG_56', 'SEEG_57', 'SEEG_58', 'SEEG_59']
raw_bip_bva.info["bads"] = ['SEEG_45', 'SEEG_46', 'SEEG_47', 'SEEG_48']

## Epoching

epochs_bip_bva = mne.Epochs(raw_bip_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3, add_eeg_ref=False)
epochs_original_bva = mne.Epochs(raw_original_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3, add_eeg_ref=False)

epochs_bip_bva['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')
epochs_original_bva['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_original = mnehelp.picks_all(epochs_original_bva)
picks_hi = mnehelp.picks_all_localised(epochs_original_bva, pd_montage, 'Hi')
box = mnehelp.custom_box_layout(picks_original, 8)
plot_picks_original = range(0, len(picks_original))

power_onset_original_bva = tfr_morlet(epochs_original_bva['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                     picks=picks_original, return_itc=False)

power_stop_perhead_bva = tfr_morlet(epochs_original_bva['stops_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                     picks=picks_original, return_itc=False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_original_bva.plot_topo(picks=plot_picks_original, baseline=(-3, -2), mode='logratio', layout=box)
