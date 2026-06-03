#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SERVER_LIST=$(python -c "from cray_infra.util.get_config import get_config; print(get_config().get('server_list', ''))" 2>/dev/null || echo "")

if [[ "$SERVER_LIST" != *"megatron"* ]] && [[ "$SERVER_LIST" != "all" ]]; then
  echo "Skipping Slurm on server_list=${SERVER_LIST}"
  exit 0
fi

# Run the slurm discovery service
python $LOCAL_DIRECTORY/../infra/cray_infra/slurm/discovery/discover_clusters.py

SLURM_CONF=${SLURM_CONF:-/app/cray/nfs/slurm.conf}
CONTROLLER=""
if [ -f "$SLURM_CONF" ]; then
  CONTROLLER=$(grep '^SlurmctldHost=' "$SLURM_CONF" | cut -d= -f2- | tr -d '[:space:]')
fi

HOSTNAME=$(hostname)
if [ -n "$CONTROLLER" ] && { [ "$HOSTNAME" = "$CONTROLLER" ] || [ "${HOSTNAME%%.*}" = "$CONTROLLER" ]; }; then
  echo "Starting slurmctld on controller node ${HOSTNAME}"
  slurmctld
else
  echo "Skipping slurmctld on ${HOSTNAME} (controller is ${CONTROLLER:-unknown})"
fi

slurmd
