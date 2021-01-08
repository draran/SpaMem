#!/Users/Shared/Scratch/Experiments/DB-SpaMem-01/envs/bin/python
'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-01-07
-----
Last Modified: 2021-01-07
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% import libraries
#===============================================================================
import pandas as pd
from pathlib import Path
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
import logging
import mne
import numpy as np
import sys
#===============================================================================
# %% format logger
#===============================================================================
log_format = '%(asctime)s\t%(filename)s\tMSG\t%(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    format=log_format,
    datefmt=date_format
)
#===============================================================================
# %% define main function
#===============================================================================
def main(ROOTPATH):
    '''
    Convert EEG data from the BVA format to the MNE raw data type.
    input:
        - ROOTPATH: the path where the eye data are stored in text format
    output:
        - success or error
    '''
    if ROOTPATH.exists():
        # list all available data sets
        eegData = sorted(ROOTPATH.glob(
            '**/EEG-P*/*.vhdr'
        ))
        for epath in eegData:
            try:
                # encode sub and run IDs
                subID = str(epath.parent).split('/')[-1].strip('EEG-P')
                # create the export folder
                EXPORPATH = (
                    ROOTPATH
                    / 'BIDS_Rawdata'
                    / 'sub-{}'.format(subID)
                    / 'eeg'
                    / 'sub-{}_task-WM_eeg-raw.fif.gz'.format(subID)
                )
                EXPORPATH.parent.mkdir(parents=True, exist_ok=True)
                # skip conversion if the data have been converted already
                if EXPORPATH.exists():
                    logging.warning('Skipping {}'.format(epath.stem))
                    continue
                # convert the eeg data
                logging.warning('Converting {}'.format(epath.stem))

                rawData = mne.io.read_raw_brainvision(epath, preload = True)
                rawData.set_montage('standard_1020')
                events, labels = mne.events_from_annotations(rawData, regexp = 'Stimulus*')
                # recode event values to match the sent triggers values
                for etype in np.unique(events[:, -1]):
                    key, val = [
                        [key, val] 
                        for key, val in labels.items()
                        if val == etype
                    ][0]
                    events[events[:, -1] == etype, -1] = int(key.split('/')[-1].strip('s'))
                # create event channel and add data to it
                infoStim = mne.create_info(
                    ['STI 014'], 
                    rawData.info['sfreq'], 
                    ['stim']
                )
                rawStim = mne.io.RawArray(
                    np.zeros((
                        1, rawData.times.size
                    )), 
                    infoStim
                )
                rawData.add_channels([rawStim], force_update_info=True)
                rawData.add_events(events)
                rawData.save(str(EXPORPATH))
            # catch and log an exception, continue with the next file
            except Exception as error:
                logging.warning(str(error))
        returnMessage = 'Done!'
    else:
        returnMessage = 'The root path cannot be found!'
    return returnMessage
#===============================================================================
# %% run conversion
#===============================================================================
if __name__ == '__main__':
    ROOTPATH = sys.argv[1]
    output = main(Path(ROOTPATH))
    logging.warning(output)
