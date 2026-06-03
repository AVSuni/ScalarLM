from cray_infra.training.slurm_jobs import sync_training_job_statuses

import logging

logger = logging.getLogger(__name__)


async def sync_training_job_status():
    try:
        sync_training_job_statuses()
    except Exception:
        logger.exception("Failed to sync training job statuses from SLURM")
