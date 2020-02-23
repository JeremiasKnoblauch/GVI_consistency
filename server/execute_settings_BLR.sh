#!/bin/bash

# Specify which settings you want to run. 
# filename_BLR: takes in the settings you run at 'EXEC' command
# filename_n: takes in the data size. Influences the 'SBATCH' preamble
# 	      as well as the 'EXEC' command

filename_BLR=BLR_settings.txt
filename_n=n.txt

# Next, loop over the entries in filename_n.
# For each n, we spawn a new simulation process
# that executes all the BLR settings in filename_BLR.

while read n; do

	# print n
	echo "$n"

	# concatenate strings to get the file name
	# for the cluster settings deployed for this n
	filename_cluster_setting="BLR_SBATCH_settings_n=${n}.txt"

	# using the file name you just constructed, 
	# change the pre-amble of the prototype file.
	# To do so, open the new file and parse through it
	# line by line.
	
	line_number=2
	
	while read sbatchline; do

		# overwrite the preamble (lines 2-4) of prototype
		# file with what is in filename_cluster_setting

		sed -i  "${line_number}s/.*/$sbatchline/" $prototype
	
        sed -i  "s/#SBATCH --nodes.*/$nodes/" $prototype
        sed -i  "s/#SBATCH --ntasks.*/$tasks_per_node/" $prototype
        sed -i  "s/#SBATCH --time.*/$time_to_run/" $prototype
	

prototype=prototype.slurm
nodes='#SBATCH --nodes=2'
tasks_per_node='#SBATCH --ntasks-per-node=28'
time_to_run='#SBATCH --time=2:00:00'

#Print the number of settings to the screen for a security check
num_settings=$(cat "${filename}" | wc -l)
echo "The number of settings is"
echo ${num_settings}
echo "The nodes, tasks per node and time allocated to each setting are"
echo ${nodes}
echo ${tasks_per_node}
echo ${time_to_run}

#Finally, loop over all the settings.
while read p; do

        #print setting
        echo "$p"

        #overwrite the preamble (lines 2-4)
        sed -i  "s/#SBATCH --nodes.*/$nodes/" $prototype
        sed -i  "s/#SBATCH --ntasks.*/$tasks_per_node/" $prototype
        sed -i  "s/#SBATCH --time.*/$time_to_run/" $prototype

        #overwritee the MY_EXEC setting by calling line number
        sed -i  "16s/.*/$p/" $prototype

        #execute the script
        sbatch $prototype

        #Be nice to the scheduler and give him a break of 1.5 seconds...
        sleep 1.5s

done < "${filename}"

