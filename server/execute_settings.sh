#!/bin/bash


#Specify which settings and the time you want to allocate for all of them
filename=AB_yacht/settings.txt
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

