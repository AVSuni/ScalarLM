from cray_infra.util.get_config import get_config
from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.main_rank_only import main_rank_only

import sys
import time

import torch

import os
import json

import logging

logger = logging.getLogger(__name__)


def _trace_harness(msg: str) -> None:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    line = f"[rank={rank}] harness [{time.monotonic():.3f}]: {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()


class TrainingHarness:
    def update_status(self, status, metadata={}):
        _trace_harness(f"update_status enter status={status} metadata={metadata}")

        current_status = get_status()

        current_status["status"] = status
        current_status["last_updated"] = time.time()
        for key, value in metadata.items():
            current_status[key] = value

        _trace_harness("update_status calling save_status")
        save_status(current_status)
        _trace_harness("update_status save_status complete")

    def checkpoint(self, checkpoint_state, checkpoint_name):
        job_config = get_job_config()

        checkpoint_path = os.path.join(job_config["job_directory"], checkpoint_name)

        torch.save(checkpoint_state, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def get_status(self):
        return get_status()


def get_status():
    try:
        with open(os.path.join(get_training_job_directory(), "status.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading job status: {e}")
        return {"status": "unknown"}


def get_training_job_directory():
    job_config = get_job_config()

    return job_config["job_directory"]

@main_rank_only
def save_status(job_status):
    _trace_harness(f"save_status enter status={job_status.get('status', '?')}")
    try:
        contents = json.dumps(job_status)
    except Exception as e:
        logger.error(f"Error serializing job status: {e}")
        return

    with open(os.path.join(get_training_job_directory(), "status.json"), "w") as f:
        f.write(contents)
