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
#===============================================================================
# %% load the idf data (converted to csv format)
#===============================================================================
ROOTPATH = Path('/Users/uqdrange/Experiments/DB-SpaMem-01')
iData = sorted(ROOTPATH.glob("**/*Samples.txt"))
with open(iData[0], 'r') as f:
    tmp = f.readlines()
    SMPS = [
        line.strip().split('\t') 
        for line in tmp 
        if 'SMP' in line or 'Time' in line

    ]
    smps_df = pd.DataFrame(
        data=SMPS[1:],
        columns=SMPS[0][:-2] 
    )
    MSGS = [
        line.strip().split('\t') 
        for line in tmp 
        if 'MSG' in line and 'Format' not in line
    ]
    msgs_df = pd.DataFrame(
        data=MSGS,
        columns=['Time', 'Type', 'Trial', 'Message']
    )
# %%
