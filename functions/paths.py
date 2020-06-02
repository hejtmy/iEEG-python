import os
from functions import read_eeg as readeeg


def participant_path(base, participant):
    return os.path.join(base, participant)


def unity_alloego_path(base, participant):
    return os.path.join(base, participant, 'UnityAlloEgo')


def eeg_path(experiment_path):
    return os.path.join(experiment_path, 'EEG', 'Preprocessed')


def prep_unity_alloego_files(base, participant):
    exp_path = eeg_path(unity_alloego_path(base, participant))
    files = dict()
    files['folder'] = exp_path
    files['montage'] = {
        'original': os.path.join(exp_path, participant + '_montage.csv'),
        'bipolar': os.path.join(
            exp_path, participant + '_montage_bipolar.csv'),
        'perElectrode': os.path.join(
            exp_path, participant + '_montage_perElectrode.csv'),
        'perHeadbox': os.path.join(
            exp_path, participant + '_montage_perHeadbox.csv'),
    }
    freq = str(readeeg.get_frequency(exp_path))
    files['EEG'] = {
        'original': os.path.join(exp_path, 'prep_' + freq + '.mat'),
        'bipolar': os.path.join(exp_path, 'prep_bipolar_' + freq + '.mat'),
        'perElectrode': os.path.join(exp_path, 'prep_perElectrode_' + freq + '.mat'),
        'perHeadbox': os.path.join(exp_path, 'prep_perHeadbox_' + freq + '.mat')
    }
    files['experiment'] = {
        'player': os.path.join(exp_path, participant + '_player.csv'),
        'events_eegtime': os.path.join(
            exp_path, participant + '_unity_eegtime_.csv'),
        'events_unitytime': os.path.join(
            exp_path, participant + '_unity_unitytime.csv'),
        'events_timesinceeegstart': os.path.join(
            exp_path, participant + '_unity_timesinceeegstart.csv'),
        'onsets': os.path.join(exp_path, participant + '_onsets.csv')
    }
    return files
