# @Author: Dragan Rangelov <uqdrange>
# @Date:   03-6-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 10-12-2019
# @License: CC-BY-4.0
#===============================================================================
# %% importing libraries
#===============================================================================
%matplotlib qt5
import numpy as np
from pathlib import Path
import pandas as pd
import itertools
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.stats as smstat
import scipy
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path().cwd().parent
#===============================================================================
# %% analysing excel data
#===============================================================================
all_df = pd.read_csv(ROOTPATH / '03_Derivatives' / 'allData.txt', sep = '\t')

# bhv data
df_K = pd.melt(all_df,
               id_vars = ['sID'],
               value_vars = [cname
                             for cname in all_df.columns
                             if 'K_' in cname],
               value_name = 'K')
df_K = pd.concat([df_K,
                  pd.DataFrame(data = list(df_K['variable'].apply(lambda x: x.split('_')).values),
                               columns = ['meas', 'task', 'side'])],
                 axis = 1)
df_pU = pd.melt(all_df,
                id_vars = ['sID'],
                value_vars = [cname
                              for cname in all_df.columns
                              if 'PU_' in cname],
                value_name = 'pU')
df_pU = pd.concat([df_pU,
                  pd.DataFrame(data = list(df_pU['variable'].apply(lambda x: x.split('_')).values),
                               columns = ['meas', 'task', 'side'])],
                 axis = 1)
df_bhv = pd.merge(df_K, df_pU, on = ['sID', 'task', 'side'])
df_bhv.drop(columns = [cname for cname in df_bhv.columns
                       if ('variable' in cname) or ('meas' in cname)],
            inplace = True)
# diffusion parameters
df_SLF1_FA = pd.melt(all_df,
                     id_vars = ['sID'],
                     value_vars = [cname
                                   for cname in all_df.columns
                                   if ('_FA' in cname) and ('SLF1_' in cname)],
                     value_name = 'SLF1_FA')
df_SLF1_FA = pd.concat([df_SLF1_FA,
                        pd.DataFrame(data = list(df_SLF1_FA['variable'].apply(lambda x: x.split('_')).values),
                                     columns = ['area', 'side', 'meas'])],
                       axis = 1)

df_SLF2_FA = pd.melt(all_df,
                     id_vars = ['sID'],
                     value_vars = [cname
                                   for cname in all_df.columns
                                   if ('_FA' in cname) and ('SLF2_' in cname)],
                     value_name = 'SLF2_FA')
df_SLF2_FA = pd.concat([df_SLF2_FA,
                        pd.DataFrame(data = list(df_SLF2_FA['variable'].apply(lambda x: x.split('_')).values),
                                     columns = ['area', 'side', 'meas'])],
                       axis = 1)

df_SLF1_MD = pd.melt(all_df,
                     id_vars = ['sID'],
                     value_vars = [cname
                                   for cname in all_df.columns
                                   if ('_MD' in cname) and ('SLF1_' in cname)],
                     value_name = 'SLF1_MD')
df_SLF1_MD = pd.concat([df_SLF1_MD,
                        pd.DataFrame(data = list(df_SLF1_MD['variable'].apply(lambda x: x.split('_')).values),
                                     columns = ['area', 'side', 'meas'])],
                       axis = 1)

df_SLF2_MD = pd.melt(all_df,
                     id_vars = ['sID'],
                     value_vars = [cname
                                   for cname in all_df.columns
                                   if ('_MD' in cname) and ('SLF2_' in cname)],
                     value_name = 'SLF2_MD')
df_SLF2_MD = pd.concat([df_SLF2_MD,
                        pd.DataFrame(data = list(df_SLF2_MD['variable'].apply(lambda x: x.split('_')).values),
                                     columns = ['area', 'side', 'meas'])],
                       axis = 1)
df_dti = pd.merge(pd.merge(df_SLF1_FA, df_SLF2_FA, on = ['sID', 'side']),
                  pd.merge(df_SLF1_MD, df_SLF2_MD, on = ['sID', 'side']),
                  on = ['sID', 'side'])
df_dti.drop(columns = [cname for cname in df_dti.columns
                       if ('variable' in cname) or ('meas' in cname) or ('area' in cname)],
            inplace = True)
df_dti_ipsi = df_dti.replace({'side': {'L': 'left', 'R': 'right'}})
df_dti_contra = df_dti.replace({'side': {'L': 'right', 'R': 'left'}})

# long format
long_df = pd.merge(df_bhv, df_dti_ipsi, on = ['sID', 'side'])
long_df = pd.merge(long_df, df_dti_contra, on = ['sID', 'side'],
                   suffixes = ['_ipsi',
                               '_contra'])
long_df.replace({'task': {'Mixed': 'Ori',
                          'spatial': 'Loc',
                          'nonspatial': 'Avg'}},
                inplace = True)
long_df.to_csv(ROOTPATH / '03_Derivatives' / 'allData_long.tsv',
               sep = '\t', index = False)
#===============================================================================
# %% LME modelling
#===============================================================================
chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)
def lrtest(llmin, llmax):
    lr = 2 * (llmax - llmin)
    p = chisqprob(lr, 1) # llmax has 1 dof more than llmin
    return lr, p

ss = long_df.copy()
# z-scoring
for cname in [cname for cname in ss.columns if cname not in ['sID', 'side', 'task']]:
    ss[cname] = scipy.stats.zscore(ss[cname])
ss.to_csv(ROOTPATH / '03_Derivatives' / 'allData_long_zscored.tsv',
          sep = '\t', index = False)
# Kappa values
lme_K = [smf.mixedlm('K ~ 1', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + SLF1_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + SLF1_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + SLF2_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + SLF2_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + SLF1_MD_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra + SLF2_MD_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
         smf.mixedlm('K ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra + task * SLF2_MD_ipsi', data = ss, groups = ss['sID']).fit(reml = False)]
llf_K = [mdl.llf for mdl in lme_K]
llr_K = [lrtest(llf_K[idx - 1], llf_K[idx]) for idx in range(1, len(lme_K))]
sig_K = np.array(lme_K[1:])[smstat.multitest.fdrcorrection(np.array(llr_K)[:, 1])[0]]
[mdl.model.formula for mdl in sig_K]
lme_K[3].summary()

# fitted guesses
lme_pU = [smf.mixedlm('pU ~ 1', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + SLF1_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + SLF1_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + SLF2_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + SLF2_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + SLF1_MD_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra + SLF2_MD_ipsi', data = ss, groups = ss['sID']).fit(reml = False),
          smf.mixedlm('pU ~ task + side + task * SLF1_FA_contra + task * SLF1_FA_ipsi + task * SLF2_FA_contra + task * SLF2_FA_ipsi + task * SLF1_MD_contra + task * SLF2_MD_ipsi', data = ss, groups = ss['sID']).fit(reml = False)]
llf_pU = [mdl.llf for mdl in lme_pU]
llr_pU = [lrtest(llf_pU[idx - 1], llf_pU[idx]) for idx in range(1, len(lme_pU))]
sig_pU = np.array(lme_pU[1:])[smstat.multitest.fdrcorrection(np.array(llr_pU)[:, 1])[0]]
[mdl.model.formula for mdl in sig_pU]
lme_pU[1].summary()
import matplotlib.pyplot as plt
plt.plot(ss.loc[:, 'SLF1_FA_contra'], ss.loc[:, 'K'], 'o', mfc = 'gray', mec = 'none')
plt.figure()
task = 'Loc'
plt.plot(ss.loc[ss['task'] == task, 'SLF2_FA_ipsi'],
         ss.loc[ss['task'] == task, 'K'],
         'o', mfc = 'none', mec = 'red')
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
# %%
allCoefs = []
for sno in bhv_df['subjectID'].unique():
    idx_sno = bhv_df['subjectID'] == sno
    for task in bhv_df['task'].unique():
        idx_task = bhv_df['task'] == task
        for cue in bhv_df['cue'].unique():
            idx_cue = bhv_df['cue'] == cue
            pred_cols = ['_'.join(i) for i in itertools.product({'ts': ['loc'],
                                                                 'ns': ['ori'],
                                                                 's': ['ori']}[task],
                                                                ['c', 'u'],
                                                                ['A', 'B', 'C'])]
            crit = bhv_df.loc[idx_sno & idx_task & idx_cue, 'response'].values
            pred = bhv_df.loc[idx_sno & idx_task & idx_cue, pred_cols].values
            allCoefs += [np.array(complexRegression(crit[:,None], pred))]

coef_df = pd.DataFrame(data= list(itertools.product(bhv_df['subjectID'].unique(),
                                                    bhv_df['task'].unique(),
                                                    bhv_df['cue'].unique(),
                                                    ['c', 'u'],
                                                    ['A', 'B', 'C'])),
                        columns = ['sNo', 'task', 'side', 'cued', 'stim'])
# here we append the length of the regression coefficient to other data (np.abs)
coef_df['abs_Theta'] = np.abs(np.array(allCoefs).reshape(-1, 1))
coef_df['cos_Theta'] = np.cos(np.angle(np.array(allCoefs).reshape(-1, 1)))
coef_df['weighted_Theta'] = coef_df['abs_Theta'] * coef_df['cos_Theta']
gav_coef = coef_df.groupby(['task', 'cued', 'stim', 'side']).mean().reset_index()

# # %% comparing analyses
# coef_df['task'] = coef_df['task'].astype('category')
# coef_df['task'].cat.reorder_categories(['s', 'ns', 'ts'], inplace = True)
# coef_df.sort_values(by = ['task','side', 'sNo'], inplace = True)
# tmp = pd.read_csv(ROOTPATH / '03_Derivatives' / 'aggWeights_corrected.tsv', sep = '\t')
# coef_df['cos_Alpha'] = tmp.loc[:,
#                                [column
#                                 for column in tmp.columns
#                                 if 'cos' in column]].stack().reset_index().loc[:, 0].values
# coef_df['abs_Alpha'] = tmp.loc[:,
#                                [column
#                                 for column in tmp.columns
#                                 if 'abs' in column]].stack().reset_index().loc[:, 0].values

# %% plotting results
import matplotlib.pyplot as plt
# non-spatial task
fig = plt.figure()
ax_NS_L = fig.add_subplot(321)
ax_NS_L.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 'ns')
                                       & (gav_coef['side'] == 'L'),
                                       'weighted_Theta'])
ax_NS_R = fig.add_subplot(322, sharex = ax_NS_L, sharey = ax_NS_L)
ax_NS_R.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 'ns')
                                   & (gav_coef['side'] == 'R'),
                                   'weighted_Theta'])

ax_S_L = fig.add_subplot(323, sharex = ax_NS_L, sharey = ax_NS_L)
ax_S_L.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 's')
                                   & (gav_coef['side'] == 'L'),
                                   'weighted_Theta'])
ax_S_R = fig.add_subplot(324, sharex = ax_NS_L, sharey = ax_NS_L)
ax_S_R.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 's')
                                   & (gav_coef['side'] == 'R'),
                                   'weighted_Theta'])

ax_TS_L = fig.add_subplot(325, sharex = ax_NS_L, sharey = ax_NS_L)
ax_TS_L.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 'ts')
                                   & (gav_coef['side'] == 'L'),
                                   'weighted_Theta'])
ax_TS_R = fig.add_subplot(326, sharex = ax_NS_L, sharey = ax_NS_L)
ax_TS_R.bar(np.arange(6), gav_coef.loc[(gav_coef['task'] == 'ts')
                                   & (gav_coef['side'] == 'R'),
                                   'weighted_Theta'])
fig.text(2.5, .7, 'Left side', transform = ax_NS_L.transData,
         ha = 'center')
fig.text(2.5, .7, 'Right side', transform = ax_NS_R.transData,
         ha = 'center')

ax_NS_L.set_ylim(-.1, .6)
ax_NS_L.set_yticks([0, .5])
ax_NS_L.set_yticklabels(['0', '.5'])
