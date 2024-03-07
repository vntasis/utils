#!/usr/bin/bash
#$ -N my_job
#$ -t 1-34
#$ -tc 10
#$ -q my_queue
#$ -cwd
#$ -l virtual_free=20G,h_rt=10:00:00
#$ -pe smp 2
#$ -V
#$ -e /my_home/logs/$TASK_ID.err
#$ -o /my_home/logs/$TASK_ID.out

#=======================================================
# This script is an example/template for submitting a
# job array to an SGE/UGE cluster
#=======================================================
# Description of parameters declared above
# -N Job_name
# -t range of array
# -tc maximum number of tasks run concurrently
# -q queue to submit
# -cwd set as working directory the current one
# -l resources requested
# -pe smp request more threads in the same node
#    for each job in the array
# -V Specifies all active environment variables
#    to be exported to the context of the job
# -e file to save standard error
# -o file to save standard output
#-------------------------------------------------------

# Set the environment
module load Python/3.6.4-foss-2018a
source $home/virtenv_py3/bin/activate

# Set home directory
home="/my/home/dir/"

# Define a list - Each item of the list will be utilized in a different job
my_var=$(ls -1 $home/samples_to_analyze/ | sed -n "${SGE_TASK_ID}p")

# Run your analysis
my_analysis.py "$my_var"

# Send an email notification at the end of running the job array
if [ "$SGE_TASK_ID" -eq "$SGE_TASK_LAST" ]
then
    echo "Job Array: $JOB_ID ($JOB_NAME) ran $SGE_TASK_LAST tasks and finished" | /bin/mail -s "Last Task: $SGE_TASK_LAST, current task $SGE_TASK_ID" my_email@domain.com
fi
