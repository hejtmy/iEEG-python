import mne
import numpy as np
import pandas as pd

from functions import mne_prepping as mneprep
from functions import mne_helpers as mnehelp
from functions import read_eeg as readeegr
from functions import mne_stats as mnestats
from functions import paths

from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:\OneDrive\FGU\iEEG\Data'



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
