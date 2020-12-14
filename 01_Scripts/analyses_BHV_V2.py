# @Author: Dragan Rangelov <uqdrange>
# @Date:   03-6-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 03-6-2019
# @License: CC-BY-4.0
#===============================================================================
# %% importing libraries
#===============================================================================
%matplotlib qt5
import numpy as np
from pathlib import Path
import pandas as pd
import itertools
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path().cwd().parent
#===============================================================================
# %% reading in behavioural data
#===============================================================================
bhv_df = pd.read_csv(ROOTPATH / '02_Rawdata' / 'BHV.tsv.gz',
                     sep = '\t')
bhv_df = bhv_df.loc[bhv_df['fixated'] == 1]
#===============================================================================
# %% running regression analysis
#===============================================================================
def complexRegression(crit, pred):
    '''
    Compute regression coefficients for predictors
    args
    ====
    crit - dependent variable, N trials x 1, -pi to pi
    pred - independent variables, N trials x M predictors, -pi to pi
    returns
    =======
    vector of coefficients 1 X M
    '''
    pred = np.exp(pred * 1j)
    crit = np.exp(crit * 1j)
    coefs = (np.asmatrix(np.asmatrix(pred).H
                        * np.asmatrix(pred)).I
             * (np.asmatrix(pred).H
                * np.asmatrix(crit)))
    return coefs

allCoefs = []
for sno in bhv_df['subjectID'].unique():
    idx_sno = bhv_df['subjectID'] == sno
    for task in bhv_df['task'].unique():
        idx_task = bhv_df['task'] == task
        pred_cols = ['_'.join(i) for i in itertools.product({'ts': ['loc'],
                                                             'ns': ['ori'],
                                                             's': ['ori']}[task],
                                                            ['c', 'u'],
                                                            ['A', 'B', 'C'])]
        crit = bhv_df.loc[idx_sno & idx_task, 'response'].values
        pred = bhv_df.loc[idx_sno & idx_task, pred_cols].values
        allCoefs += [np.array(complexRegression(crit[:,None], pred))]

sNo, task, cued, stim = np.array(list(itertools.product(bhv_df['subjectID'].unique(),
                  bhv_df['task'].unique(),
                  ['c', 'u'],
                  ['A', 'B', 'C']))).T
coef_df = pd.DataFrame(data= list(itertools.product(bhv_df['subjectID'].unique(),
                                                    bhv_df['task'].unique(),
                                                     ['c', 'u'],
                                                     ['A', 'B', 'C'])),
                        columns = ['sNo', 'task', 'cued', 'stim'])

# here we append the length of the regression coefficient to other data (np.abs)
coef_df['coefs'] = np.abs(np.array(allCoefs).reshape(-1, 1))
gav_coef = coef_df.groupby(['task', 'cued', 'stim']).mean().reset_index()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.bar(np.arange(6), gav_coef.loc[gav_coef['task'] == 'ns', 'coefs'])

fig = plt.figure()
plt.bar(np.arange(6), gav_coef.loc[gav_coef['task'] == 's', 'coefs'])

fig = plt.figure()
plt.bar(np.arange(6), gav_coef.loc[gav_coef['task'] == 'ts', 'coefs'])
