import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = "D:\\IntracranialElectrodes\\Data\\p126\\"
#base_path = "U:\\OneDrive\\FGU\\iEEG\\p126\\"

path_montage = base_path + "UnityAlloEgo\\EEG\\Preprocessed\\p126_montage.csv"

path_perhead_bva = "prep_perHeadbox_250.mat"
path_bip_bva =  base_path + "prep_bipolar_250.mat"
path_events_bva = "base_path + p126_BVA500_1500.csv"

FREQUENCY = 250

pd_montage = readeegr.read_montage(path_montage)

# loading BVA data
raw_perhead_bva = mneprep.load_raw(path_perhead_bva, FREQUENCY)
raw_bip_bva = mneprep.load_raw(path_bip_bva, FREQUENCY)
mne_events_bva, mapp_bva = mneprep.load_events(path_events_bva, FREQUENCY)

# {'c': 1, 'f': 2, 'g': 3, 'onsets_500_1500': 4, 'stops_500_1500': 5}
raw_perhead_bva.plot(events=mne_events_bva, scalings='auto',
                     event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})
raw_bip_bva.plot(events=mne_events_bva, scalings='auto',
                 event_color={1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'})

raw_perhead_bva.info["bads"] = ['SEEG_55', 'SEEG_56', 'SEEG_57', 'SEEG_58', 'SEEG_59']
raw_bip_bva.info["bads"] = ['SEEG_45', 'SEEG_46', 'SEEG_47', 'SEEG_48']

## Epoching

epochs_bip_bva = mne.Epochs(raw_bip_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3, add_eeg_ref=False)
epochs_perhead_bva = mne.Epochs(raw_perhead_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3, add_eeg_ref=False)

epochs_bip_bva['onsets_500_1500', 'stops_500_1500'].plot(block=True, scalings='auto')

freqs = np.arange(2, 30, 1)
n_cycles = freqs / 2

picks_original = mnehelp.def_picks(epochs_bip_bva)
box = mnehelp.custom_box_layout(picks_original, 8)
plot_picks_perhead = range(0, len(picks_original))

power_onset_perhead_bva = tfr_morlet(epochs_perhead_bva['onsets_500_1500'], freqs=freqs, n_cycles=n_cycles,
                                     picks=picks_original, return_itc=False)
# NEED to pass picks because default IGNORES SEEG channels
power_onset_perhead_bva.plot_topo(picks=plot_picks_perhead, baseline=(-2., -1.5), mode='logratio', layout=box)