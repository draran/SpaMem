'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2020-12-15
-----
Last Modified: 2020-12-15
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2020 Dragan Rangelov, The University of Queensland
'''
# Find all IDF data on the server, and copy these files to one folder
# so that all files could be converted in one go
#===============================================================================
# %% import necessary libraries
#===============================================================================
from pathlib import Path
import shutil
#===============================================================================
# %% move IDF data from RDM folder to Shared folder
#===============================================================================
ROOTPATH = Path('/Volumes/COSMOS2018-I1002')
# check if the RDM folder has been mounted
if ROOTPATH.exists():
    DESTPATH = Path('/Users/uqdrange/Shared/eyeData_IDF')
    if not DESTPATH.exists():
        DESTPATH.mkdir(parents=True, exist_ok=True)
    # list collected idf files
    pathsIDF = sorted(ROOTPATH.glob('**/*.idf'))
    # list already converted files
    pathsCSV = sorted(ROOTPATH.glob('**/*Samples.txt'))
    # select only unconverted files
    sourcePaths = [
        path 
        for path in pathsIDF 
        if path.parent / ' '.join([path.stem, 'Samples.txt']) not in pathsCSV
    ]
    # copy source files to shared folder accessible to virtual machine
    for source in sourcePaths:
        # skip if file has already been copied
        if not (DESTPATH / source.name).exists():
            shutil.copy(
                source,
                DESTPATH / source.name
            )
else:
    print('The RDM network folder is not mounted!')
#===============================================================================
# %% move converted data from Shared folder to RDM folder
#===============================================================================
# the location of the converted data 
CSVPATH = Path('/Users/uqdrange/Shared/eyeData_CSV')
if not CSVPATH.exists():
    CSVPATH.mkdir(parents=True, exist_ok=True)
# iterate over the original IDF files
for source in sourcePaths:
    # set the path of the converted CSV file
    dest = source.parent / ' '.join([source.stem, 'Samples.txt'])
    # skip copying if the file has been copied already
    if not dest.exists():
        shutil.copy(
            CSVPATH / dest.name,
            dest
        )
#===============================================================================
# %% housekeeping - remove local original and converted files
#===============================================================================
def rm(dir_path, fun):
    try:
        fun(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
rm(DESTPATH, shutil.rmtree)
rm(CSVPATH, shutil.rmtree)