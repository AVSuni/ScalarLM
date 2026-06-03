from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.launch_training_job import start_slurm_job
from cray_infra.training.slurm_jobs import (
    get_active_slurm_jobs,
    is_slurm_job_active,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.util.get_config import get_config


import traceback
import os
import json
import yaml
import time

import logging

logger = logging.getLogger(__name__)


async def restart_megatron_jobs():
    logger.info("Checking for Megatron jobs to restart")

    active_slurm = get_active_slurm_jobs()
    logger.info("Active SLURM job names: %s", list(active_slurm.keys()))

    async for job in get_restartable_jobs():
        job_name = os.path.basename(job)
        if job_name in active_slurm:
            logger.info("Skipping restart for %s: already in squeue", job_name)
            continue
        await restart_job(job)

    active_slurm = get_active_slurm_jobs()
    if active_slurm:
        logger.info("Jobs still running in SLURM, keeping the server alive")
        await keep_alive()


async def get_restartable_jobs():
    config = get_config()
    max_restarts = config.get("max_job_restart_count", 3)
    cooldown = config.get("job_restart_cooldown_seconds", 300)

    if not os.path.exists(config["training_job_directory"]):
        return

    for path in os.listdir(config["training_job_directory"]):
        root = os.path.join(config["training_job_directory"], path)
        status_path = os.path.join(root, "status.json")
        if not os.path.exists(status_path):
            continue
        try:
            with open(status_path) as f:
                status = json.load(f)
        except Exception as e:
            logger.error("Error reading status.json for job %s: %s", root, e)
            traceback.print_exc()
            continue

        if status.get("status") != TrainingJobStatus.QUEUED:
            continue

        restart_count = status.get("restart_count", 0)
        if restart_count >= max_restarts:
            logger.warning(
                "Job %s exceeded max restart count (%s), marking FAILED",
                path,
                max_restarts,
            )
            _mark_job_failed(
                root,
                status,
                f"Exceeded max restart count ({max_restarts})",
            )
            continue

        last_restart = status.get("last_restart_at", 0)
        if time.time() - last_restart < cooldown:
            continue

        yield root


def _mark_job_failed(job_directory: str, status: dict, reason: str) -> None:
    status["status"] = TrainingJobStatus.FAILED
    status["error"] = reason
    status["last_updated"] = time.time()
    status_path = os.path.join(job_directory, "status.json")
    with open(status_path, "w") as f:
        json.dump(status, f)


async def restart_job(job):
    logger.info("Restarting job: %s", job)

    status_path = os.path.join(job, "status.json")
    with open(status_path) as f:
        status = json.load(f)

    status["restart_count"] = status.get("restart_count", 0) + 1
    status["last_restart_at"] = time.time()
    with open(status_path, "w") as f:
        json.dump(status, f)

    with open(os.path.join(job, "config.yaml")) as f:
        config = yaml.safe_load(f)

    job_name = os.path.basename(job)
    if is_slurm_job_active(job_name):
        logger.info("SLURM job %s became active before restart, skipping sbatch", job_name)
        return

    start_slurm_job(config)


async def keep_alive():
    config = get_config()
    session = get_global_session()
    try:
        async with session.get(config["api_url"] + "/v1/health/keepalive") as resp:
            assert resp.status == 200
    except Exception as e:
        logger.error(f"Error keeping the server alive: {e}")
