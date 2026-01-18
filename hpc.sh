#!/bin/bash

#submit_and_wait --partition=gprod_gssi -N 1 --ntasks-per-node=3 --gres=gpu:a100:8 hpc_cluster_run.sh 
# scontrol show job 42973

# Get user jobs
# squeue -u $USER

# Initialize (optional - function handles empty LAST_JOB_ID)
unset LAST_JOB_ID

user=$(whoami)

# Get list of current jobs for the user
jobs=$(squeue -u "$user" -h -o "%A")

if [ -z "$jobs" ]; then
    echo "No jobs found for user $user."
fi

read -p "Do you want to delete all current jobs? [y/N]: " answer

case "$answer" in
    [Yy]* )
        rm *.log
        rm *.err
        rm *.out
        for job in $jobs; do
            scancel "$job"
            echo "Removed job: $job"
        done
        ;;
    * )
        echo "No jobs were deleted."
        ;;
esac

wait_for_job() {
    local jobid=$1
    echo "Waiting for job $jobid to finish..."
    while squeue -j "$jobid" &>/dev/null; do
        sleep 60  # check every 60 seconds
        echo "Job $jobid running..."
    done
    echo "Job $jobid finished."
}

submit_and_wait_old() {
    local max_jobs=2
    local jobid

    # Wait until fewer than max_jobs of our submitted jobs are running
    while [ "$(squeue -u "$(whoami)" -h -o "%A" | wc -l)" -ge "$max_jobs" ]; do
        echo "Max $max_jobs jobs running. Waiting..."
        sleep 60
    done

    # Submit the job
    jobid=$(sbatch "$@" | awk '{print $4}')
    echo "Submitted job $jobid"
}

submit_and_wait() {  
    local jobid
    # If there's a previous job, add dependency
    if [[ -n "$LAST_JOB_ID" ]]; then
        jobid=$(sbatch --dependency=afterok:$LAST_JOB_ID "$@" | awk '{print $4}')
        echo "Submitted job $jobid (waiting for job $LAST_JOB_ID)"
    else
        jobid=$(sbatch "$@" | awk '{print $4}')
        echo "Submitted job $jobid (no dependency)"
    fi
    
    # Update the last job ID for next submission
    LAST_JOB_ID=$jobid
    
    # Return the job ID for potential use
    echo $LAST_JOB_ID
}

#################################
##        Classical ANN        ##
#################################
bash hpc_ann.sh

#################################
## The same tasks but with KAN ##
#################################
bash hpc_kan.sh

#####################################
## The same tasks but with Koopman ##
#####################################
bash hpc_koopman.sh

###########################################
## The same tasks but with KAN + Koopman ##
###########################################
bash hpc_kan_koopman.sh

###################################
## The same tasks but with MAMBA ##
###################################
# bash hpc_mamba.sh

###################################
## The same tasks but with Mixed ##
###################################
#bash hpc_koopman_mixed.sh

###########
## TESTS ##
###########
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh SpacecraftCWAffine "5 6 1 6 10 1 0 0"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:h100:1 hpc_cluster_run.sh SpacecraftCWAffine "5 6 1 6 10 1 0 0 1"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:h100:1 hpc_cluster_run.sh AUV "5 7 1 6 10 1 0 0"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV "5 7 1 6 10 1 0 0"
#submit_and_wait --partition=gprod -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh AUV_A100 "5 7 1 6 10 1 0 0"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:h100:1 hpc_cluster_run.sh AUV_H100 "5 7 1 6 10 1 0 0"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh AUV_KAN_A100 "5 7 1 6 10 1 0 0 1"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:h100:1 hpc_cluster_run.sh AUV_KAN_H100 "5 7 1 6 10 1 0 0 1"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_KAN "5 7 1 6 10 1 0 0 1"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh SpacecraftCWAffine "5 6 1 6 10 1 0 0"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh SpacecraftCWAffine "5 6 1 6 10 1 0 0 1"
#submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0"