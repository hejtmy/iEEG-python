# Oscillations Go the Distance: Low-Frequency Human Hippocampal Oscillations Code Spatial Distance in the Absence of Sensory Cues during Teleportation

# Lindsay K. Vass, Milagros S. Copara, Masud Seyal, Kiarash Shahlaie, Sarah Tomaszewski Farias, Peter Y. Shen, and Arne D. Ekstrom2
# https://www.cell.com/cms/10.1016/j.neuron.2016.01.045/attachment/6f25e416-ed11-4ea5-bbaa-e3dd769fb1d0/mmc1
import mne
import numpy as np

from functions import mne_prepping as mneprep
from functions import mne_loading as loader
from functions import paths
from functions import mne_analysis as mneanalysis

# from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

base_path = 'E:/OneDrive/FGU/iEEG/Data'
participant = 'p136'
scalings = {'seeg': 1e2, 'ecg': 1e2, 'misc': 1e2}

file_paths = paths.prep_unity_alloego_files(base_path, participant)
eeg, montage = loader.load_eeg(file_paths, 'bipolar')

pd_events = mneprep.load_preprocessed_events(file_paths)
mne_events, events_mapp = mneprep.pd_to_mne_events(pd_events, eeg.info['sfreq'])

# Preprocessing
eeg.notch_filter(50)

# We quantified oscillatory episodes using the method of Caplan et al. (2001, 2003). This approach identifies epochs of iEEG signal that show high-oscillatory power at a particular frequency lasting several cycles. The method excludes much of the background noise signal by estimating the noise spectrum. A minimum-duration threshold helps to exclude spikes, sharpwaves, and nonrhythmic changes in power. The analysis is performed separately at each frequency of interest.

# For a given frequency, f*, an oscillatory episode is defined as an epoch longer than a duration threshold, DT(in numbers of cycles), during which wavelet power at frequency f* exceeded a power threshold, PT. 

# We wavelet-transformed the raw iEEG signal [Morlet wavelet, window 6 cycles,] at 22 logarithmically spaced frequencies in the range 1–32 Hz to obtain the wavelet power spectrum.

#  We assumed that thebackground noise spectrum has the form Power(f)¼Afa. We estimated this background by fitting the observed spectrum(at each electrode) with a linear regression in log-log units. Because wavelet power values are expected to be distributed likev Chi2, the estimated background at f* should be the mean of its corresponding chi2 distribution. e chose PT(f) to be the 95th percentile of the fit distribution. Power thresholding should exclude about 95% of the 

# We set DTto three cycles off, or 3/f. This was done to eliminate artifacts and nonrhythmicphysiological signals.

# Finally, Pepisode(f), or the percentageof time in oscillatory episodes, was defined as the total amoun tof time filled with detected oscillatory episodes divided by thetotal time in the segment of interest. I