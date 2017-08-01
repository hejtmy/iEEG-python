import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr

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
pick_perhead_hip = mnehelp.picks_all_localised(raw_perhead_bva, pd_montage_referenced, 'Hi')
pick_perhead_hip_names = mne.pick_info(raw_perhead_bva.info, pick_perhead_hip)['ch_names']
pick_perhead_ins = mnehelp.picks_all_localised(raw_perhead_bva, pd_montage_referenced, 'Ins')
pick_perhead_all = mnehelp.picks_all(raw_perhead_bva)

## Epoching
epochs_perhead_bva = mne.Epochs(raw_perhead_bva, mne_events_bva, event_id=mapp_bva, tmin=-3, tmax=3)
bad_epochs_perhead = [19, 51, 52, 185, 186, 204, 205, 287, 288, 289, 302, 367, 368, 373, 374, 375, 376, 377, 378, 418, 419, 488, 489, 490, 504, 505, 506, 507]
epochs_perhead_vr.drop(bad_epochs_perhead)
#epochs_perhead_vr.plot(block = True, scalings = 'auto', picks=pick_perhead_hip)


## TIME FREQUENCY
freqs = np.arange(2, 10, 1)
n_cycles = freqs / 2

picks_perhead_hi = mnehelp.picks_all_localised(epochs_perhead_bva, pd_montage, 'Hi')
box = mnehelp.custom_box_layout(picks_original, 8)
plot_picks_original = range(0, len(picks_original))



# NEED to pass picks because default IGNORES SEEG channels
power_onset_original_bva.plot_topo(picks=plot_picks_original, baseline=(-3, -2), mode='logratio', layout=box)
