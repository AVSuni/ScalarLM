#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# u unset variables are errors
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuo pipefail
if [[ "${CRAY_TRAIN_DEBUG:-0}" == "1" ]]; then
  set -x
fi

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"
export LOCAL_DIRECTORY

# Job ml overrides container; infra comes from Dockerfile PYTHONPATH.
export PYTHONPATH=$LOCAL_DIRECTORY/ml:$PYTHONPATH

export PYTHONUNBUFFERED=1

NODEFILE=$(mktemp)
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$NODEFILE"
NUM_NODES=$(wc -l < "$NODEFILE")

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Ensure essential env vars are set/exported in the api pod. Suggested example for single process per node:
# export NCCL_IB_GID_INDEX=1
# export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export HSA_NO_SCRATCH_RECLAIM=1
# export NCCL_SOCKET_IFNAME=eth0
# export GLOO_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_NET_GDR_LEVEL=SYS
# export ROCR_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=0

echo "[launcher t=$(date +%s)] LOCAL_DIRECTORY=${LOCAL_DIRECTORY}" >&2
echo "[launcher t=$(date +%s)] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-?} SLURM_NTASKS=${SLURM_NTASKS:-?} SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-?}" >&2
echo "[launcher t=$(date +%s)] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE}" >&2
echo "[launcher t=$(date +%s)] hostfile:" >&2
cat "$NODEFILE" >&2

ulimit -l unlimited
mpirun --allow-run-as-root \
  --hostfile "$NODEFILE" \
  -np "$NUM_NODES" \
  -npernode $SLURM_NTASKS_PER_NODE \
  --bind-to none \
  bash -c '
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export LOCAL_WORLD_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE

echo "Launching torchrun rank=$RANK local_rank=$LOCAL_RANK world=$WORLD_SIZE local_world_size $LOCAL_WORLD_SIZE on $(hostname) master=$MASTER_ADDR:$MASTER_PORT" >&2

ulimit -l unlimited

torchrun \
  --nnodes="$WORLD_SIZE" \
  --nproc-per-node=$LOCAL_WORLD_SIZE \
  --node_rank="$RANK" \
  --local-rank="$LOCAL_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$LOCAL_DIRECTORY/ml/cray_megatron/main.py" "$@"
' _ "$@"

MPIRUN_EXIT=$?
rm -f "$NODEFILE"
echo "[launcher t=$(date +%s)] mpirun finished exit=${MPIRUN_EXIT}" >&2
exit "$MPIRUN_EXIT"