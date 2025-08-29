#!/bin/bash

#submit_and_wait --partition=gprod_gssi -N 1 --ntasks-per-node=3 --gres=gpu:a100:8 hpc_cluster_run.sh 
# scontrol show job 42973

# Initialize (optional - function handles empty LAST_JOB_ID)
unset LAST_JOB_ID

rm *.log
rm *.err
rm *.out

# Get all job IDs for the current user and cancel them
user=$(whoami)
for job in $(squeue -u "$user" -h -o "%A"); do
    scancel "$job"
done

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
submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF2Affine "2 1 1 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine "5 1 1 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF2NonAffine "2 1 1 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5Affine "5 1 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF2Affine "2 1 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine "5 1 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF2NonAffine "2 1 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF5Affine "5 2 1 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF2Affine "2 2 1 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF5NonAffine "5 2 1 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF2NonAffine "2 2 1 6 10 0 0 0"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF2Affine "2 3 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF2NonAffine "2 3 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF5Affine "5 3 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF5NonAffine "5 3 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF2Affine "2 4 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF2NonAffine "2 4 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF5Affine "5 4 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF5NonAffine "5 4 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF2Affine "2 5 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF2NonAffine "2 5 0 6 10 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF5NonAffine "5 5 0 6 10 0 0 0"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoState "5 3 0 6 10 0 1 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoInput "5 3 0 6 10 0 2 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NonAffine2 "5 3 0 2 2 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NonAffine4 "5 3 0 4 4 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoState "5 1 0 6 10 0 1 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoInput "5 1 0 6 10 0 2 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine2 "5 1 0 2 2 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine4 "5 1 0 4 4 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoState "5 4 0 6 10 0 1 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoInput "5 4 0 6 10 0 2 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine2 "5 4 0 2 2 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine4 "5 4 0 4 4 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine24 "5 4 0 2 4 0 0 0"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoState "5 1 1 6 10 0 1 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoInput "5 1 1 6 10 0 2 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine25 "5 1 1 2 5 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine3 "5 1 1 3 3 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine5 "5 1 1 5 5 0 0 0"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine35 "5 1 1 3 5 0 0 0"

#################################
## The same tasks but with KAN ##
#################################
submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB --gres=gpu:a100:1 hpc_cluster_run.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF2Affine "2 1 1 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine "5 1 1 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF2NonAffine "2 1 1 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5Affine "5 1 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF2Affine "2 1 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine "5 1 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF2NonAffine "2 1 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF5Affine "5 2 1 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF2Affine "2 2 1 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF5NonAffine "5 2 1 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh HWSystemF2NonAffine "2 2 1 6 10 0 0 0 1"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF2Affine "2 3 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF2NonAffine "2 3 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF5Affine "5 3 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetRHF5NonAffine "5 3 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF2Affine "2 4 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF2NonAffine "2 4 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF5Affine "5 4 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetTankNLF5NonAffine "5 4 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF2Affine "2 5 0 6 10 1 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF5NonAffine "5 5 0 6 10 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh datasetSilverNLF2NonAffine "2 5 0 6 10 0 0 0 1"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoState "5 3 0 6 10 0 1 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoInput "5 3 0 6 10 0 2 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NonAffine2 "5 3 0 2 2 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh RHReducedF5NonAffine4 "5 3 0 4 4 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoState "5 1 0 6 10 0 1 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoInput "5 1 0 6 10 0 2 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine2 "5 1 0 2 2 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankLinF5NonAffine4 "5 1 0 4 4 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoState "5 4 0 6 10 0 1 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoInput "5 4 0 6 10 0 2 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine2 "5 4 0 2 2 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine4 "5 4 0 4 4 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh DSTankReducedF5NonAffine24 "5 4 0 2 4 0 0 0 1"

# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoState "5 1 1 6 10 0 1 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoInput "5 1 1 6 10 0 2 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine25 "5 1 1 2 5 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine3 "5 1 1 3 3 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine5 "5 1 1 5 5 0 0 0 1"
# submit_and_wait --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh NLTankNLF5NonAffine35 "5 1 1 3 5 0 0 0 1"
