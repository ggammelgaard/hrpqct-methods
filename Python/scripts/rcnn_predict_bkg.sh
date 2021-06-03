#!/bin/bash

# We need unet folder and cnn subfolder
if [ $# -eq 0 ]
    then
        echo "2 arguments needed: 'unet_subfolder' and 'cnn_subfolder'"
        exit
fi
if [ $# -eq 1 ]
    then
        echo "2 arguments needed: 'unet_subfolder' and 'cnn_subfolder'"
        exit
fi
unet_subfolder=$1
cnn_subfolder=$2

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`

# Change conda environment
command2="source activate 3dunet_36"

# Start script
command3="python $MAINPATH/main_master.py rcnn_predict $unet_subfolder $cnn_subfolder"

# Run commands
echo "Saving output to folder: $unet_subfolder"
eval $command1
eval $command2
eval $command3
