#!/Users/Shared/Scratch/Experiments/DB-SpaMem-01/envs/bin/python
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
import logging
import mne
import numpy as np
import sys
# %% plotting libraries DONE: comment this out if running jobs
# import matplotlib as mpl
# mpl.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()
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
# %% testing parameters 
# DONE: comment these out
#===============================================================================
# ROOTPATH = Path('/Volumes/COSMOS2018-I1002')
# ipath = (
#     ROOTPATH 
#     / 'Raw_data_all' 
#     / 'Behav_Eye' 
#     / 'P01' 
#     / 'Subject1_Session1_EYE__block_1 Samples.txt'
# )
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
        logging.warning('Finding all relevant files to convert')
        # list all available data sets
        iData = sorted(ROOTPATH.glob(
            "**/Behav_Eye/P*/Subject*Samples.txt"
        ))
        for ipath in iData:
            try:
                # encode sub and run IDs
                subID = str(ipath.parent).split('/')[-1].strip('P').rjust(3,'0')
                runID = ipath.stem.split('_')[-1].split(' ')[0].rjust(2, '0')
                # create the export folder
                EXPORPATH = (
                    ROOTPATH
                    / '02_Rawdata'
                    / 'sub-{}'.format(subID)
                    / 'eye'
                    / 'sub-{}_task-WM_run-{}_eye-raw.fif.gz'.format(subID, runID)
                )
                EXPORPATH.parent.mkdir(parents=True, exist_ok=True)
                # skip conversion if the data have been converted already
                if EXPORPATH.exists():
                    logging.warning('Skipping {}'.format(ipath.stem))
                    continue
                # convert the eye data
                logging.warning('Converting {}'.format(ipath.stem))
                # load the eye data
                with open(ipath, 'r', encoding='utf-8') as f:
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
                # get the sample data
                smps_df = pd.DataFrame(
                    data=[
                        line.strip().split('\t') 
                        for line in tmp
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
                smps_df['Timestamp'] = pd.to_datetime(
                    smps_df['Time'],
                    unit='us'
                )
                # get the message data
                msgs_df = pd.DataFrame(
                    data=[
                        line.strip().split('\t') 
                        for line in tmp
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
                msgs_df['Description'] = msgs_df['Message'].apply(
                    lambda x: x.split(': ')[-1]
                )
                msgs_df['Timestamp'] = pd.to_datetime(
                    msgs_df['Time'],
                    unit='us'
                )
                msgs_df['Timedelta'] = (
                    msgs_df['Timestamp'] 
                    - smps_df['Timestamp'][0]
                )
                msgs_df['Onsets'] = msgs_df['Timedelta'] / np.timedelta64(1, 's')
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
                        smps_df[[
                            'L Raw X [px]', 
                            'L Raw Y [px]', 
                            'L Mapped Diameter [mm]'
                        ]].values.T,
                        np.zeros(smps_df.shape[0])[None]
                    ]),
                    rawInfo
                )
                # create annotations from messages
                rawAnnotations = mne.Annotations(
                    onset=msgs_df['Onsets'],
                    duration=.0001,
                    description=msgs_df['Description']
                )
                rawData.set_annotations(rawAnnotations)
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