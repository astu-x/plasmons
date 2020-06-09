#!/bin/env bash

#first part to execute, with job id saved
k_points1=25
for ((i=0;i<=$k_points1;i++));
do
    cmd_line='jobid'$i'=$(sbatch --parsable run.sh '$i')'
    eval $cmd_line
done

#Second part to execute, with job dependency, to regulate amount of cores used
k_points2=50
for ((i=$((k_points1+1));i<=$k_points2;i++));
do
    cmd_line='sbatch --dependency=afterok:$jobid'$(($i - $k_points1 - 1))' run.sh '$i
    eval $cmd_line
done
