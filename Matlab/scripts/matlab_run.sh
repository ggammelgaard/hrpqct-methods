#!/bin/bash

# We need the matlab script to run
if [ $# -eq 0 ]
        then
                echo "Argument needed: 'script_to_run'"
                exit
fi
script_to_run=$1

matlab_command="\"run('${script_to_run}');exit;\""
command1="/mnt/data/joe/matlab2021_program_root/bin/matlab -nodisplay -nosplash -nodesktop -r ${matlab_command}"

eval $command1
