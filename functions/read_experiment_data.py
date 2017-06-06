# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:12:36 2016

@author: hejtm
"""
import pandas as pd
import numpy as np

def read_events(path):
    events = pd.read_csv(path, sep = ",", quoting = 2) #quoting 2 is QUOTE_NONNUMERIC (2)
    return(events)

"""
It is necessary to supply an “events” array in order to create an Epochs object. This is of shape(n_events, 3)
where the first column is the sample number (time) of the event, the second column indicates the value from which
the transition is made from (only used when the new value is bigger than the old one), and the third column
is the new event value.
http://martinos.org/mne/dev/auto_tutorials/plot_creating_data_structures.html
"""
