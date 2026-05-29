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

#mpirun --allow-run-as-root python $LOCAL_DIRECTORY/ml/cray_megatron/main.py $*

NODEFILE=$(mktemp)
for i in $(seq 0 $((SLURM_NTASKS - 1))); do
  echo scalarlm-megatron-$i
done > $NODEFILE

NUM_NODES=$(wc -l < $NODEFILE)

export WORLD_SIZE=$SLURM_NTASKS
export RANK=${HOSTNAME##*-}
export MASTER_ADDR=scalarlm-megatron-0.scalarlm-megatron-headless
export MASTER_PORT=29500
export NCCL_IB_GID_INDEX=1
# export NCCL_DEBUG=INFO
# export LOG_LEVEL=DEBUG
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_SOCKET_IFNAME=net1

export PYTHONUNBUFFERED=1

# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_NCCL_ENABLE_MONITORING=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=64000

# torchrun --nnodes=$SLURM_NTASKS \
#     --nproc-per-node=1  \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#      $LOCAL_DIRECTORY/ml/cray_megatron/main.py $*

ulimit -l unlimited
mpirun --allow-run-as-root \
  --hostfile $NODEFILE \
  -np $NUM_NODES -npernode 1 \
  --bind-to none \
  bash -c '
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_GID_INDEX=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export NCCL_DEBUG=INFO
# export LOG_LEVEL=DEBUG
export NCCL_SOCKET_IFNAME=net1

export PYTHONUNBUFFERED=1

# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_NCCL_ENABLE_MONITORING=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=64000

echo "Launching torchrun rank=$RANK world=$WORLD_SIZE on $(hostname)"

ulimit -l unlimited

torchrun \
  --nnodes=$WORLD_SIZE \
  --nproc-per-node=1 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $LOCAL_DIRECTORY/ml/cray_megatron/main.py "$@"
'
