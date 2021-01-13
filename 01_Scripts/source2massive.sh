# move behavioural and eye data
# for dry run add -n to options
nohup rsync -ruhP \
/Volumes/COSMOS2018-I1002/Raw_data_all/Behav_Eye \
massive:/scratch/im34/DB-SpaMem-01/00_Sourcedata/ \
>> source2massive.log &
# move eeg data
nohup rsync -ruhP \
/Volumes/COSMOS2018-I1002/Raw_data_all/EEG \
massive:/scratch/im34/DB-SpaMem-01/00_Sourcedata/ \
>> source2massive.log &

# nohup rsync -ruhPn \
# /Volumes/COSMOS2018-I1002/Raw_data_all/Scripts/WM-run \
# massive:/scratch/im34/DB-SpaMem-01/00_Sourcedata/Scripts \
# >> source2massive.log &
