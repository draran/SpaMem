'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-01-14
-----
Last Modified: 2021-01-29
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% import libraries
#===============================================================================
import numpy as np
import logging
from pathlib import Path
from numpy.core.shape_base import block
import pandas as pd
import itertools
# %% import libraries for plotting
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
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
# %% get all data
#===============================================================================
ROOTPATH = Path('/scratch/im34/DB-SpaMem-01')
bhvFiles = sorted(ROOTPATH.glob('**/sub-*_task-WM_bhv.tsv.gz'))
# %% load and concatenate all data
all_df = []
for fpath in bhvFiles:
    subNo = int(fpath.stem.split('_')[0].split('-')[-1])
    logging.warning('Loading subject number {}'.format(subNo))
    # NOTE: missing responses are indexed as 9999
    tmp_df = pd.read_csv(fpath, sep = '\t', na_values=9999)
    tmp_df['subNo'] = subNo
    all_df += [tmp_df]
all_df = pd.concat(all_df)
# %% recode task and cue: 
#   1 = spatial [s] / ori, 
#   2 = non spatial [ns] / avg, 
#   3 = true spatial [ts] / loc
all_df['task'] = all_df['cond']
all_df['task'].replace(
    [1, 2, 3],
    ['ori', 'avg', 'loc'],
    inplace=True
) 
#   1 = left, 
#   2 = right
all_df['side'] = all_df['cue']
all_df['side'].replace(
    [1, 2],
    ['left', 'right'],
    inplace=True
) 
# %% recode angles 
for col in range(1, 7):
    all_df[f'P_{col}_rad'] = np.arctan2(
        *all_df.loc[:, [f'Y_{col}', f'X_{col}']].values.T
    )
    # we divide the angle by 90 cause these are orientations
    all_df[f'A_{col}_rad'] = all_df[f'A_{col}'].values * np.pi / 90
    # normalize the range of orientations to -pi - pi
    all_df[f'A_{col}_rad'] = np.angle(np.exp(all_df[f'A_{col}_rad'] * 1j))
# %% recode angles
# NOTE: for location task, participants were instructed to report only one side 
# so it may not be justified to consider location as a 360 degrees task, 
# but rather a 180 degrees task.
[
    all_df['tarRad'],
    all_df['rspRad']
] =  (all_df[['targetAngle', 'responseAngle']] * np.pi / 90).values.T
# for location task we need to divide by 180 to get to pirad, 
# since we have divided by 90 already, we just need to divide by 2
all_df.loc[all_df['task'] == 'loc', ['tarRad', 'rspRad']] /= 2
# wrap angles
all_df[['tarRad','rspRad']] = np.angle(np.exp(all_df[['tarRad','rspRad']] * 1j))
# compute error magnitude
all_df['errRad'] = np.angle(
    np.exp(all_df['tarRad'] * 1j) 
    / np.exp(all_df['rspRad'] * 1j)
)
# %% rearange columns depending on the cued side 
tar_df = pd.concat([
    all_df.loc[
        (all_df['side'] == side)
        & (all_df['task'] == task),
        [
            col 
            for col in all_df.columns
            if ('rad' in col) 
            and  (
                (('A' in col) and (task != 'loc'))
                or (('P' in col) and (task == 'loc'))
            )
            and (
                ((side == 'left') and (int(col.split('_')[1]) % 2 != 0))
                or ((side == 'right') and (int(col.split('_')[1]) % 2 == 0)) 
            )
        ]
    ].rename(
        columns=dict(
            zip(
                [
                    col 
                    for col in all_df.columns
                    if ('rad' in col) 
                    and  (
                        (('A' in col) and (task != 'loc'))
                        or (('P' in col) and (task == 'loc'))
                    )
                    and (
                        ((side == 'left') and (int(col.split('_')[1]) % 2 != 0))
                        or ((side == 'right') and (int(col.split('_')[1]) % 2 == 0)) 
                    )
                ],
                [
                    '_'.join(col)
                    for col in itertools.product(['TAR'], ['1', '2', '3'])
                ]
            )
        )
    )
    for side in all_df['side'].unique()
    for task in all_df['task'].unique()
])
dis_df = pd.concat([
    all_df.loc[
        (all_df['side'] == side)
        & (all_df['task'] == task),
        [
            col 
            for col in all_df.columns
            if ('rad' in col) 
            and  (
                (('A' in col) and (task != 'loc'))
                or (('P' in col) and (task == 'loc'))
            )
            and (
                ((side == 'left') and (int(col.split('_')[1]) % 2 == 0))
                or ((side == 'right') and (int(col.split('_')[1]) % 2 != 0)) 
            )
        ]
    ].rename(
        columns=dict(
            zip(
                [
                    col 
                    for col in all_df.columns
                    if ('rad' in col) 
                    and  (
                        (('A' in col) and (task != 'loc'))
                        or (('P' in col) and (task == 'loc'))
                    )
                    and (
                        ((side == 'left') and (int(col.split('_')[1]) % 2 == 0))
                        or ((side == 'right') and (int(col.split('_')[1]) % 2 != 0)) 
                    )
                ],
                [
                    '_'.join(col)
                    for col in itertools.product(['DIS'], ['1', '2', '3'])
                ]
            )
        )
    )
    for side in all_df['side'].unique()
    for task in all_df['task'].unique()
])
tmp_df= pd.concat([
    tar_df,
    dis_df
], axis = 1).sort_index()
all_df.drop(
    columns=[
    col 
    for col in all_df.columns
    if '_' in col
], inplace=True)
all_df = all_df.merge(
    tmp_df,
    left_index=True, right_index=True
)
# %% remove missing data
all_df.dropna(inplace = True)   
# remove bad trials
all_df = all_df.loc[all_df['trialOK'] == 1]
#===============================================================================
# %% analyse error magnitudes
#===============================================================================
# compute bins for histograms
NBINS = 10
binIntervals= np.linspace(-np.pi, np.pi, NBINS)
binLabels = np.linspace(-np.pi, np.pi, NBINS + 1)[1:-1]
all_df['errBin'] = pd.cut(all_df['errRad'], binIntervals, labels=binLabels)
# compute frequencies per bin
binFreqs = all_df.groupby([
    'subNo', 
    'task', 
    'cue', 
    'errBin'
]).size().reset_index().rename(columns={0:'binFreq'})
binFreqs['binProp'] = binFreqs.groupby([
    'subNo',
    'task',
    'cue'        
]).apply(lambda x: x['binFreq']/ x['binFreq'].sum()).reset_index()['binFreq']
gavBins = binFreqs.groupby([
    'task',
    'cue',
    'errBin'
]).mean().reset_index()
# %% plot empirical distributions of error magnitudes
fig = plt.figure()
for idx_task, task in enumerate(['ori', 'loc', 'avg']):
    ax = plt.subplot(1, 3, idx_task + 1)
    for cue in [1, 2]:
        ax.bar(
            binLabels, 
            gavBins.loc[
                (gavBins['task'] == task)
                & (gavBins['cue'] == cue),
                'binProp'
            ],
            alpha = .3,
            width = .6
        )
    ax.set_title(task)
    ax.set_ylim(0, .6)
# %%
# TODO: 
# 2. characterise empirical distribution of error magnitudes (SD and M) 
# 3. Mixture distribution model fitting: 
#   a.  fixed mu = 0; same K 
#   b.  variable swap coefficients for cued and uncued side 
#   c.  for averaging task allow variable K for target and swaps, 
#       perhaps penalize if Ktarget > K swaps 
#   d. compute decision weights
