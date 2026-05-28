#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"

# Put the current ml directory in the python path so that the modules can be imported
export PYTHONPATH=$LOCAL_DIRECTORY/ml:$PYTHONPATH

export WORLD_SIZE=${SLURM_NTASKS:-1}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-net1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export PYTHONUNBUFFERED=1

srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 bash -c '
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
python '"$LOCAL_DIRECTORY"'/ml/cray_megatron/main.py "$@"
' -- "$@"
