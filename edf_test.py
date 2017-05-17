import mne
import numpy as np

path = "D:\\IntracranialElectrodes\\MatlabScripts\\iEEG_Analysistest.edf"
data = mne.io.read_raw_edf(path)