#!/bin/bash

# We need input id
if [ $# -eq 0 ]
    then
        echo "Arguments needed: kfold_subfolder"
        exit
fi
kfold_subfolder=$1

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`

# Change conda environment
command2="source activate 3dunet_36"

# Start script
command3="python $MAINPATH/main_master.py cnn_convert $kfold_subfolder"

# Run commands
echo "Saving output to folder: $kfold_subfolder"
eval $command1
eval $command2
eval $command3
