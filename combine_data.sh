#!/bin/env bash


#create string to combine to output.dat and execute
cmd_line1="cat "
cmd_line2="rm "
k_points=50
for ((i=0;i<=$k_points;i++));
do
    cmd_line1=$cmd_line1'data/output'$i'.dat '
	cmd_line2=$cmd_line2'data/output'$i'.dat '
done
cmd_line1=$cmd_line1' > data/output.dat'
eval $cmd_line1
eval $cmd_line2

#create string to combine to raw_field data.dat and execute

#1
cmd_line1="cat "
cmd_line2="rm "
k_points=50
for ((i=0;i<=$k_points;i++));
do
    cmd_line1=$cmd_line1'data/raw_field1_'$i'.dat '
	cmd_line2=$cmd_line2'data/raw_field1_'$i'.dat '
done
cmd_line1=$cmd_line1' > data/raw_field1.dat'
eval $cmd_line1
eval $cmd_line2


#1_lcp
cmd_line1="cat "
cmd_line2="rm "
k_points=50
for ((i=0;i<=$k_points;i++));
do
    cmd_line1=$cmd_line1'data/raw_field1_lcp_'$i'.dat '
	cmd_line2=$cmd_line2'data/raw_field1_lcp_'$i'.dat '
done
cmd_line1=$cmd_line1' > data/raw_field1_lcp.dat'
eval $cmd_line1
eval $cmd_line2