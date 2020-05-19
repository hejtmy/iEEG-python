import mne

from functions import mne_prepping as mneprep
from functions import read_eeg as readeegr
from functions import paths


base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
frequency = readeegr.get_frequency(paths.eeg_path(paths.unity_alloego_path(base_path, participant)))

# loading montage
pd_montage = readeegr.read_montage(file_paths['montage']['original']) 
pd_montage_referenced = readeegr.read_montage(file_paths['montage']['referenced']) 

# Loading Unity data
raw_original = mneprep.load_raw(file_paths['EEG']['base'], frequency, pd_montage)
raw_perhead = mneprep.load_raw(file_paths['EEG']['perHeadbox'], frequency, pd_montage_referenced)
raw_perelectrode = mneprep.load_raw(file_paths['EEG']['perElectrode'], frequency, pd_montage_referenced)
raw_bipolar = mneprep.load_raw(file_paths['EEG']['bipolar'], frequency, pd_montage_referenced)

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, frequency)

# Epoching
epochs_original = mne.Epochs(raw_original, mne_events, event_id=events_mapp,
                             tmin=-3, tmax=3)
epochs_original.plot(scalings=scalings)

# Playing
raw_original.plot(scalings=scalings)
raw_original.plot_psd(fmax=100, picks=['seeg'], average=False)

raw_perhead.plot(scalings=scalings)
raw_perelectrode.plot(scalings=scalings)
raw_bipolar.plot(scalings=scalings)

raw_bipolar.plot(events=mne_events, color='gray', scalings=scalings)