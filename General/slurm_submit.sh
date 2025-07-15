#!/usr/bin/bash
#SBATCH --job-name my_job
#SBATCH --output=/my_home/logs/%x.%A_%a.out
#SBATCH --error=/my_home/logs/%x.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --array=0-11%3

#=======================================================
# This script is an example/template for submitting a
# job array to a Slurm cluster
#=======================================================

set -euo pipefail

# Start message #
#---------------#
start_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] starting on $(hostname)

# Set the environment #
#---------------------#
# module load my_module
module load Python/3.11.5-GCCcore-13.2.0
# conda activate my_env

# Set working directory #
#-----------------------#
work_dir="/my/work/dir"

# Define a list #
#---------------#
# Each item of the list will be utilized in a different job #
my_file=$(\
    ls -1 "$work_dir/samples_to_analyze/" | \
    sed -n "$((SLURM_ARRAY_TASK_ID+1))p"
)

# Run your analysis #
#-------------------#
my_analysis.py "$my_file"

# End message #
#-------------#
cgroup_dir=$(awk -F: '{print $NF}' /proc/self/cgroup)
peak_mem=`cat /sys/fs/cgroup$cgroup_dir/memory.peak`
echo [$(date +"%Y-%m-%d %H:%M:%S")] peak memory is $peak_mem bytes
end_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
