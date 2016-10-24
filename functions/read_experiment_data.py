# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:12:36 2016

@author: hejtm
"""
import pandas as pd

def read_events(path):
    events = pd.read_csv(path, sep = ",", quoting = 2) #quoting 2 is QUOTE_NONNUMERIC (2)
    return(events)