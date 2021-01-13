#!/Users/uqdrange/anaconda2/envs/mne/bin/python
'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2020-12-10
-----
Last Modified: 2020-12-10
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2020 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% import necessary libraries
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
    Convert eye data from text format to MNE raw data type.
    input:
        - ROOTPATH: the path where the eye data are stored in text format
    output:
        - success or error
    '''
    if ROOTPATH.exists():
        # list all available data sets
        iData = sorted(ROOTPATH.glob(
            "Raw_data_all/Behav_Eye/P*/Subject*Samples.txt"
        ))
        for ipath in iData:
            try:
                # encode sub and run IDs
                subID = str(ipath.parent).split('/')[-1].strip('P')
                runID = ipath.stem.split('_')[-1].split(' ')[0].rjust(2, '0')
                # create the export folder
                EXPORPATH = (
                    ROOTPATH
                    / 'BIDS_Rawdata'
                    / 'sub-{}'.format(subID)
                    / 'eye'
                    / 'sub-{}_task-WM_run-{}_eye.fif.gz'.format(subID, runID)
                )
                EXPORPATH.parent.mkdir(parents=True, exist_ok=True)
                # skip conversion if the data have been converted already
                if EXPORPATH.exists():
                    logging.warning('Skipping {}'.format(ipath.stem))
                    continue
                # convert the eye data
                logging.warning('Converting {}'.format(ipath.stem))
                # NOTE: the first message per recording run is "StartBlock_"
                # NOTE: we will export only the recorded data for a given run 
                with open(ipath, 'r', encoding='utf-8') as f:
                    # load the eye data
                    tmp = f.readlines()
                    # save the meta data
                    iInfo = dict([
                        (entry.split(':\t')[0], entry.split(':\t')[1]) 
                        for entry in [
                            line.strip().strip('## ') for line in tmp
                            # header lines are marked with ##
                            if '##' in line
                        ]
                        if ':\t' in entry
                    ])
                    idx_start = [
                        idx 
                        for idx, line in enumerate(tmp)
                        if 'StartBlock_{}'.format(int(runID)) in line
                    ]
                    # check if there were more than one starting messages
                    if len(idx_start) > 1:
                        raise IndexError
                    else:
                        idx_start = idx_start[0]
                    # get the sample data
                    smps_df = pd.DataFrame(
                        data=[
                            line.strip().split('\t') 
                            for line in tmp[idx_start:] 
                            if 'SMP' in line

                        ],
                        columns=[
                            line.strip().split('\t')
                            for line in tmp
                            if '##' not in line
                            and 'SMP' not in line
                            and 'MSG' not in line
                        # this will result in only one element
                        # and the last two col names are not needed
                        ][0][:-2]
                    # select only the needed columns
                    )[[
                        'Time',
                        'L Raw X [px]',
                        'L Raw Y [px]',
                        'L Mapped Diameter [mm]'
                    # specify data types
                    ]].astype({
                        'Time':'int64',
                        'L Raw X [px]':'float',
                        'L Raw Y [px]':'float',
                        'L Mapped Diameter [mm]':'float'
                    })
                    # get the message data
                    msgs_df = pd.DataFrame(
                        data=[
                            line.strip().split('\t') 
                            for line in tmp[idx_start:]
                            # Select only messages sent to the tracker 
                            if 'Message' in line
                            # # Select only messages sent by the experiment code
                            # and '_' in line
                            # # Remove block related messages
                            # and 'StartBlock' not in line
                        ],
                        columns=['Time', 'Type', 'Trial', 'Message']
                    )[[
                        'Time',
                        'Message'
                    ]].astype({
                        'Time':'int64',
                        'Message':'str'
                    })
                    # encode block type and the event separately
                    [
                        msgs_df['blockType'],
                        msgs_df['event']
                    ] = zip(*[
                        msg.split(': ')[-1].split('_') 
                        for msg in msgs_df['Message']
                    ])
                    # recode the events using eeg trigger values
                    msgs_df = msgs_df.replace(
                        to_replace=dict(
                            blockType=dict(
                                Spatial=1,
                                NonSpatial=2,
                                TrueSpatial=3
                            ),
                            event=dict(
                                StartCue=10,
                                StartEnc=20,
                                StartMaint=30
                            )
                        ),

                    )
                    msgs_df['Trig'] = msgs_df['blockType'] + msgs_df['event']
                # %% create event array
                idx_trig = np.argmin(
                    np.abs(
                        msgs_df['Time'].values[:, None] 
                        - smps_df['Time'].values[None]
                    ), 1
                ) 
                events = np.stack([
                    idx_trig,
                    np.zeros(idx_trig.size),
                    msgs_df['Trig'].values
                ], 1).astype('int')
                # %% create raw array of idata
                rawInfo = mne.create_info(
                    [
                        'L Raw X [px]',
                        'L Raw Y [px]',
                        'L Diameter [mm]',
                        'STI 014'
                    ],
                    int(iInfo['Sample Rate']),
                    ch_types=[
                        'misc',
                        'misc',
                        'misc',
                        'stim'
                    ]
                )
                rawInfo['iView opts'] = iInfo
                rawData = mne.io.RawArray(
                    np.concatenate([
                        smps_df.values.T[1:],
                        np.zeros(smps_df.shape[0])[None]
                    ]),
                    rawInfo
                )
                # TODO: create annotations instead of events
                rawData.add_events(events)
                rawData.save(str(EXPORPATH))
            # catch and log an exception, continue with the next file
            except Exception as error:
                logging.warning(str(error))
        returnMessage = 'Done!'
    else:
        returnMessage = 'The remote folder is not mounted!'
    return returnMessage
#===============================================================================
# %% run conversion
#===============================================================================
if __name__ == '__main__':
    ROOTPATH = Path(sys.argv[1])
    output = main(ROOTPATH)
    logging.warning(output)
