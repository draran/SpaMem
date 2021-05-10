# configure paths and variables
CONDAPATH=/projects/im34/miniconda/bin/conda
PROJECT=im34
ENV=DB-SpaMem-01
ROOTPATH="/scratch/${PROJECT}/${ENV}"

#nohup $CONDAPATH env create \
#--verbose \
#--file "${ROOTPATH}/01_Scripts/mne_env.yml" \
#--prefix "${ROOTPATH}/env" \
#--force \
#> "${ROOTPATH}/create_environment.out" &

# create environment
nohup \
$CONDAPATH create \
--prefix "${ROOTPATH}/env" \
--override-channels \
--channel conda-forge \
--strict-channel-priority \
--no-default-packages \
--verbose \
--yes \
python">=3.8" \
pip \
numpy \
scipy \
matplotlib \
numba \
pandas \
xlrd \
scikit-learn \
h5py \
pillow \
statsmodels \
jupyter \
joblib \
psutil \
numexpr \
traits \
pyface \
traitsui \
imageio \
tqdm \
imageio-ffmpeg">=0.4.1" \
vtk">=9.0.1" \
pyvista">=0.24" \
pyvistaqt">=0.2.0" \
mayavi \
PySurfer \
dipy \
nibabel \
nilearn \
python-picard \
pyqt \
mne \
mffpy">=0.5.7" \
> "${ROOTPATH}/create_environment.log" &

echo "DONE"


