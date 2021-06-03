#!/bin/bash

# We need kfold subfolder
if [ $# -eq 0 ]
    then
        echo "Arguments needed: cnn_subfolder"
        exit
fi

cnn_subfolder=$1

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`
output_path="$MAINPATH/output_cnn/$cnn_subfolder/predict"
command1="mkdir -p $output_path"

# Change conda environment
command2="source activate 3dunet_36"

# Start script
command3="python $MAINPATH/main_master.py cnn_predict $cnn_subfolder"

# Run commands
echo "Saving output to folder: $cnn_subfolder"
eval $command1
eval $command2
eval $command3
