#!/bin/bash

# We need input id
if [ $# -eq 0 ]
    then
        echo "Arguments needed: kfold_subfolder"
        exit
fi
kfold_subfolder=$1

# Decide on folder name
now=$(date +'%y%m%d_%H%M')

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`
output_path="$MAINPATH/output_unet/$now/train"
command1="mkdir -p $output_path"

# Change conda environment
command2="source activate 3dunet_36"

# Start script
command3="nohup python -u $MAINPATH/main_master.py unet_train $kfold_subfolder $now > $output_path/console_train.txt &"

# Run commands
echo "Saving output to folder: $now"
eval $command1
eval $command2
eval $command3
