#!/projects/im34/miniconda/envs/spamem_env/bin/python
'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-01-13
-----
Last Modified: 2021-01-13
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% importing packages
#===============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import json
import logging
from scipy.io import loadmat
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
# ROOTPATH = Path('/Volumes/COSMOS2018-I1002') #DONE: comment this out
def main(ROOTPATH):
    '''
    Convert BHV data from the MAT format to the CSV data.
    input:
        - ROOTPATH: the path where the eye data are stored in text format
    output:
        - success or error
    '''
    if ROOTPATH.exists():
        # list all available data sets
        logging.warning('Finding all relevant files to convert')
        bhvData = sorted(ROOTPATH.glob(
            '**/Behav_Eye/P*/Subject*.mat'
        ))
        # NOTE: for some subject there are more than a single output file
        # get unique participant IDs
        subjects = np.unique([
            str(fpath.parent).split('/')[-1] 
            for fpath in bhvData
        ])
        for sub in subjects:
            try:
                # encode subID
                subID = sub.strip('P').rjust(3,'0')
                logging.warning('Converting data for sub-{}'.format(subID))
                # create the export folder
                EXPORPATH = (
                    ROOTPATH
                    / '02_Rawdata'
                    / 'sub-{}'.format(subID)
                    / 'beh'
                    / 'sub-{}_task-WM_bhv.tsv.gz'.format(subID)
                )
                EXPORPATH.parent.mkdir(parents=True, exist_ok=True)
                # DONE: uncomment this 
                # skip conversion if the data have been converted already
                if EXPORPATH.exists():
                    logging.warning('Skipping {}'.format(fpath.stem))
                    continue
                # get all the files
                fpaths = np.array([
                    fpath 
                    for fpath in bhvData 
                    if sub in str(fpath)
                ])
                # sort the data in ascending order of creation data
                fpaths = fpaths[np.argsort([
                    fpath.stat().st_ctime
                    for fpath in fpaths
                ])]
                logging.warning(
                    '{} file/s found for conversion.'.format(len(fpaths))
                )
                # NOTE: matData is a dictionary comprising of the following:
                #   /data - all trials in the order of presentation, ntrls x 11
                #       blocknumber, blockID, trialcount,
                #       cond, cue, targetangle
                #       gaborangle_polar, responseangle_polar,
                #       deltatheta (error magnitude), resptime,
                #       goodtrials
                #   /<task>/run_data -  all trials of per task, ntrls_task x 8
                #       icond, icue, 
                #       target_angle, 
                #       circle_location_angle,
                #       circletargetpositionX, circletargetpositionY, 
                #       screentargetpositionX, screentargetpositionY
                #   /<task>/run_data_X - ntrials_task x 6
                #       X coordinates [px] for the six presented gabor patches 
                #       [0:3] - Left side; [3:] - Right side 
                #       [0] - Left target, [3] - Right target
                #   /<task>/run_data_Y - ntrials_task x 6
                #       Y coordinates, conventions as for X
                #   /<task>/run_data_A - ntrials_task x 6
                #       Angle from vertical, conventions as for X
                # NOTE: different conditons' labels
                # 1 = spatial [s], 2 = non spatial [ns], 3 = true spatial [ts]
                bhv_df_all = []
                for fpath in fpaths: 
                    matData = loadmat(fpath, simplify_cells=True)['OUTPUT']
                    x_centre, y_centre = [
                        matData['screen']['widthpixel'] * .5,
                        matData['screen']['heightpixel'] * .5
                    ]
                    bhv_df = pd.DataFrame(
                        data=matData['data'],
                        columns=[
                            'blockNo',
                            'blockID',
                            'trialNo',
                            'cond',
                            'cue',
                            'patchAngle',
                            'targetAngle',
                            'responseAngle',
                            'deltaAngle',
                            'RT',
                            'trialOK',
                        ]
                    ).astype(
                        dtype=dict(
                            blockNo='int',
                            blockID='int',
                            trialNo='int',
                            cond='int',
                            cue='int',
                            patchAngle='float',
                            targetAngle='float',
                            responseAngle='float',
                            deltaAngle='float',
                            RT='float',
                            trialOK='int',
                        )
                    ).drop(
                        # we don't need this one
                        # this is the angle value for the element 1 or 2 
                        # depending on the cued side
                        columns=['patchAngle']
                    )
                    # add info about presented stimuli per trial
                    stimInfo_cnames = [
                        '_'.join(comb) 
                        for comb in product(
                            ['X','Y','A'],
                            ['1','3','5','2','4','6']
                        )
                    ]
                    bhv_df[stimInfo_cnames] = np.nan
                    # add stim info only for the conditions that have been run
                    for cond in bhv_df['cond'].unique():
                        task = ['s', 'ns', 'ts'][cond - 1]
                        bhv_df.loc[
                            bhv_df['cond'] == cond,
                            stimInfo_cnames
                        ] = pd.DataFrame(
                                data=np.concatenate([
                                    matData[task]['run_data_X'],
                                    matData[task]['run_data_Y'],
                                    matData[task]['run_data_A']
                                ], -1),
                                columns=stimInfo_cnames
                        ).values
                    
                    # center coordinates to the screen centre
                    for col in range(1, 7):
                        bhv_df[f'X_{col}'] -= x_centre
                        bhv_df[f'Y_{col}'] -= y_centre
                    
                    bhv_df_all += [bhv_df]
                # concatenate all bhv data into one data frame
                bhv_df = pd.concat(
                    bhv_df_all, 
                    # keep track which data come from which file
                    # NOTE: trials are sorted relative to creation times 
                    # with lower indices reflecting earlier times
                    keys=[
                        'file_{}'.format(idx_path)
                        for idx_path in range(len(bhv_df_all))
                    ])
                bhv_df.to_csv(
                    EXPORPATH,
                    sep='\t',
                    na_rep='n/a'
                )
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
if __name__ == "__main__":
    ROOTPATH = Path(sys.argv[1])
    output = main(ROOTPATH)
    logging.warning(output)