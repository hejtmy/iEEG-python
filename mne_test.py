import mne
import numpy

ch_names = list(range(1, 51))
ch_names = [str(ch) for ch in ch_names]
ch_types = ["seeg"] * 50
info = mne.create_info(ch_names = ch_names, sfreq = 250, ch_types = ch_types)
raw = mne.io.RawArray(bip_data, info)