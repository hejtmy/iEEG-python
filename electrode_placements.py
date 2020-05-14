from scipy.io import loadmat
import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from os import path as op

import mne

print(__doc__)

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
path_data = mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat'

# We've already clicked and exported
layout_path = op.join(op.dirname(mne.__file__), 'data', 'image')
layout_name = 'custom_layout.lout'
