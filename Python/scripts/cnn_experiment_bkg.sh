#!/bin/bash

# We need py script and kfold subfolder
if [ $# -eq 0 ]
    then
        echo "2 arguments needed: 'experiment_to_run' and 'kfold_subfolder'"
        exit
fi
if [ $# -eq 1 ]
    then
        echo "2 arguments needed: 'experiment_to_run' and 'kfold_subfolder'"
        exit
fi
experiment_to_run=$1
kfold_subfolder=$2

# Decide on folder name
now=$(date +'%y%m%d_%H%M')
experiment_folder="${now}_c$experiment_to_run"

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`
output_path="$MAINPATH/output_cnn/$experiment_folder"
command1="mkdir -p $output_path"

# Change conda environment
command2="source activate 3dunet_36"

# Start script
command3="nohup python -u $MAINPATH/cnn_experiments/$experiment_to_run.py $kfold_subfolder $experiment_folder > $output_path/console.txt &"

# Run commands
echo "Saving output to folder: $experiment_folder"
eval $command1
eval $command2
eval $command3
