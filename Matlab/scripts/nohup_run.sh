#!/bin/bash

# We need the matlab script to run
if [ $# -eq 0 ]
        then
                echo "Argument needed: 'script_to_run'"
                exit
fi
script_to_run=$1

# Decide on log name
now=$(date +'%y%m%d_%H%M%S')

# Make directory to add nohup output log
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MAINPATH=`dirname $SCRIPTPATH`
output_path="$MAINPATH/output/logs"
command1="mkdir -p $output_path"

nohup_log="${output_path}/${now}_nohup.txt"
matlab_command="\"run('${script_to_run}');exit;\""
command2="nohup /mnt/data/joe/matlab2021_program_root/bin/matlab -nodisplay -nosplash -nodesktop -r ${matlab_command} > ${nohup_log} &"

eval $command1
eval $command2

echo "Running ${script_to_run} as nohup"
echo "Saving log to path: ${nohup_log}"
