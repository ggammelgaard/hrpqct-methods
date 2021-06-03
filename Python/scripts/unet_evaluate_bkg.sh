#!/bin/bash

# We need input id
if [ $# -eq 0 ]
    then
        echo "Arguments needed: unet_subfolder"
        exit
fi

unet_subfolder=$1

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`
output_path="$MAINPATH/output_unet/$unet_subfolder/evaluate"
command1="mkdir -p $output_path"

# Change conda environment
command2="source activate 3dunet_36"

# Start script
#command3="nohup python -u $MAINPATH/main_unet_evaluate.py $unet_subfolder > $output_path/console_evaluate.txt &"
command3="python $MAINPATH/main_master.py unet_evaluate $unet_subfolder"

# Run commands
echo "Saving output to folder: $unet_subfolder"
eval $command1
eval $command2
eval $command3
