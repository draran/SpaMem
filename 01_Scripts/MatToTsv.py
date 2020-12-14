# @Author: Dragan Rangelov <uqdrange>
# @Date:   14-2-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 03-6-2019
# @License: CC-BY-4.0
# %%
#===============================================================================
# importing packages
#===============================================================================
import numpy as np
import pandas as pd
import pathlib
import json
from itertools import product
# %%
#===============================================================================
# read matlab data to pandas
#===============================================================================
def mat2pd(path, data):
    '''
    Read mat file as a python data structure
    :Params
        - path - location of the mat file
        - data - which part of the mat file should be read in
    :Returns
        - dictionary with keys being data structures stored in data
    '''
    import scipy.io as spio
    with path.open('rb') as f:
        mat = spio.loadmat(f)
    names = mat[data].dtype. names
    ndata = {name: mat[data][name][0, 0] for name in names}
    return ndata

#===============================================================================
# wrapping functions
#===============================================================================

def wrapTopi(theta):
    '''
    Wrap array of angles in pi radians from -pi to pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + np.pi) % (2 * np.pi) - np.pi

def wrapTo2pi(theta):
    '''
    Wrap array of angles in pi radians from 0 to 2pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + 2 * np.pi) % (2 * np.pi)

# %%
#===============================================================================
# setting paths and collecting mat files
#===============================================================================
ROOTPATH = '/Users/uqdrange/Experiments/DB-SpaMem-01'
SOURCEPATH = ROOTPATH + '/00_Sourcedata'

sourcePath = pathlib.Path(SOURCEPATH)
matFiles = sorted(list(sourcePath.glob('*.mat')))
bad_data = ['s05'] # s05 did not complete enough trials for cond == 3
matFiles = [mfile for mfile in matFiles if mfile.name[:3] not in bad_data]
#===============================================================================
with open(ROOTPATH + '/01_Scripts/output.json', 'r') as f:
    output = json.load(f)
datacolumns = list(output['data']['columns'].keys())

# %%
# DONE: for s08, s20, s29 1:1 join fails
all_BHV_data = pd.DataFrame()
mergeFailed = []
for outputfile in matFiles:
    sno = outputfile.name.split('.')[0]
    tmp = mat2pd(outputfile, 'OUTPUT')
    # reading in recorded data
    tmp_df = pd.DataFrame(tmp['data'],
                          columns = datacolumns)
    # adding subject id column
    tmp_df.insert(loc = 0, column = 'subjectID', value = sno)
    # recoding conditions so that its easier to manipulate them
    tmp_df['task'].replace(
        [1, 2, 3],
        ['s', 'ns', 'ts'],
        inplace = True)
    tmp_df['cue-side'].replace(
        [1, 2],
        ['L', 'R'],
        inplace = True)
    # dropping columns that will be recomputed
    tmp_df.drop(['cond', 'targetAngle', 'targetAngle_polar', 'deltaPolar'],
                axis = 1,
                inplace = True)
    tmp_df.rename(columns = {'responseAngle_polar': 'response',
                             'cue-side': 'cue'},
                  inplace = True)
    # re-aranging column order
    tmp_df = tmp_df[['subjectID',
                     'blockNo',
                     'trialNo',
                     'task',
                     'cue',
                     'response',
                     'RT',
                     'fixated']]
    tmp_df['blockNo'], tmp_df['trialNo'] = tmp_df[['blockNo','trialNo']].astype('int').values.T
    # recoding the response to radians -pi to pi
    idx_ori = tmp_df['task'].isin(['s', 'ns'])
    tmp_df.loc[idx_ori, 'response'] = wrapTopi(tmp_df.loc[idx_ori, 'response']
                                               * np.pi / 90)
    tmp_df.loc[~idx_ori, 'response'] = wrapTopi(tmp_df.loc[~idx_ori, 'response']
                                                * np.pi / 180)
    # adding stimulus properties
    x_res, y_res = tmp['screen']['res'][0,0][0]
    tmp_df['key'] = ['_'.join([i, j, k, l, m, n])
                     for i, j, k, l, m, n in zip(*tmp_df.loc[:,
                                                    ['task',
                                                     'blockNo',
                                                     'trialNo',
                                                     'response',
                                                     'RT',
                                                     'fixated']].values.astype('str').T)]

    thetas_df = pd.DataFrame()
    for task in tmp_df['task'].unique():
        theta_Oris = np.angle(np.exp(2 * np.pi
                                     - (tmp[task]['run_data_A'][0, 0]
                                        * np.pi / 90) * 1j) # recoding angle to go CCW
                              * np.exp((90 * np.pi / 90) * 1j)) # rotating the angle for 90 degrees

        theta_Locs = np.angle(tmp[task]['run_data_X'][0, 0] - x_res / 2 +
                              # negative sign as the pixel coordinates start from upper-left
                              -(tmp[task]['run_data_Y'][0, 0] - y_res / 2) * 1j)

        # recoding left/right to cued and uncued
        attRight = tmp_df.loc[tmp_df['task'] == task, 'cue'] == 'R'
        theta_Oris[attRight] = np.concatenate([theta_Oris[attRight,3:],
                                               theta_Oris[attRight,:3]],
                                              axis = 1)
        theta_Locs[attRight] = np.concatenate([theta_Locs[attRight,3:],
                                               theta_Locs[attRight,:3]],
                                              axis = 1)

        if task == 'ns':
            oris = theta_Oris[:, :3]
            target = np.angle(np.exp(oris * 1j).sum(1))
        elif task == 's':
            target = theta_Oris[:, 0]
        elif task == 'ts':
            target = theta_Locs[:, 0]


        task_df = pd.DataFrame(data = np.concatenate([target[:, None],
                                                      theta_Oris,
                                                      theta_Locs],
                                                     axis = 1),
                               columns = ['target'] + ['_'.join(cname)
                                                       for cname
                                                       in list(product(['ori', 'loc'],
                                                                       ['c_A',
                                                                        'c_B',
                                                                        'c_C',
                                                                        'u_A',
                                                                        'u_B',
                                                                        'u_C']))])

        task_df['key'] = tmp_df.loc[tmp_df['task'] == task, 'key'].values
        thetas_df = thetas_df.append(task_df, ignore_index = True)

    try:
        tmp_df = tmp_df.merge(thetas_df, on = 'key', validate = '1:1')
    except:
        # some trials were duplicated/copied twice
        (sno, tmp_df.loc[tmp_df.duplicated(), 'trialNo'])
        mergeFailed += [(sno,
                         tmp_df.loc[tmp_df.duplicated(), 'trialNo'])]
        # s21, trials 132 and 120 were copied twice
        tmp_df.drop_duplicates(['key'], inplace = True)
        thetas_df.drop_duplicates(['key'], inplace = True)
        # appending stim info to other data
        tmp_df = tmp_df.merge(thetas_df, on = 'key', validate = '1:1')
    tmp_df.drop(['key'], axis = 1, inplace = True)

    all_BHV_data = all_BHV_data.append(tmp_df, ignore_index = True)

all_BHV_data.to_csv(ROOTPATH + '/02_Rawdata/BHV.tsv.gz',
                    sep = '\t', na_rep = 'n/a', index = False,
                    compression = 'gzip')
