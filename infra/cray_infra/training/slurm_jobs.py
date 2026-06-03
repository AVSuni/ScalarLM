from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.util.get_config import get_config

import json
import os
import subprocess
import time
import logging

from typing import Dict, Optional

logger = logging.getLogger(__name__)

# SLURM states that mean the job is still in the scheduler queue or running.
_ACTIVE_SQUEUE_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "PREEMPTED",
        "REQUEUED",
        "RESIZING",
        "REVOKED",
        "SIGNALING",
        "SPECIAL_EXIT",
        "STAGE_OUT",
    }
)

# Terminal sacct states mapped to training job status.
_SACCT_TERMINAL_STATUS = {
    "COMPLETED": TrainingJobStatus.COMPLETED,
    "FAILED": TrainingJobStatus.FAILED,
    "CANCELLED": TrainingJobStatus.FAILED,
    "CANCELED": TrainingJobStatus.FAILED,
    "TIMEOUT": TrainingJobStatus.FAILED,
    "NODE_FAIL": TrainingJobStatus.FAILED,
    "OUT_OF_MEMORY": TrainingJobStatus.FAILED,
    "PREEMPTED": TrainingJobStatus.FAILED,
    "BOOT_FAIL": TrainingJobStatus.FAILED,
    "DEADLINE": TrainingJobStatus.FAILED,
}


def get_job_name(train_args: Dict) -> str:
    return os.path.basename(train_args["job_directory"])


def get_active_slurm_jobs() -> Dict[str, Dict[str, str]]:
    """Return {job_name: {job_id, state}} for non-terminal squeue entries."""
    try:
        output = subprocess.check_output(
            ["squeue", "-h", "-o", "%i %j %T"],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            "squeue failed (exit %s): %s",
            e.returncode,
            e.output.decode("utf-8", errors="replace") if e.output else "",
        )
        return {}

    jobs: Dict[str, Dict[str, str]] = {}
    for line in output.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        job_id, job_name = parts[0], parts[1]
        state = parts[2] if len(parts) > 2 else "UNKNOWN"
        if state in _ACTIVE_SQUEUE_STATES or state == "UNKNOWN":
            jobs[job_name] = {"job_id": job_id, "state": state}
    return jobs


def is_slurm_job_active(job_name: str) -> bool:
    return job_name in get_active_slurm_jobs()


def is_slurm_job_active_for_train_args(train_args: Dict) -> bool:
    return is_slurm_job_active(get_job_name(train_args))


def cancel_slurm_jobs_for_train_args(train_args: Dict) -> None:
    job_name = get_job_name(train_args)
    status_path = os.path.join(train_args["job_directory"], "status.json")

    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                status = json.load(f)
            job_id = status.get("job_id")
            if job_id:
                _scancel_job_id(str(job_id))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read status for cancel: %s", e)

    try:
        subprocess.run(
            ["scancel", "--name", job_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as e:
        logger.error("scancel --name=%s failed: %s", job_name, e)


def _scancel_job_id(job_id: str) -> None:
    try:
        subprocess.run(
            ["scancel", job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as e:
        logger.error("scancel %s failed: %s", job_id, e)


def get_sacct_state(job_id: str) -> Optional[str]:
    try:
        output = subprocess.check_output(
            [
                "sacct",
                "-j",
                str(job_id),
                "-n",
                "-X",
                "--parsable2",
                "-o",
                "State",
            ],
            stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, OSError):
        return None

    for line in output.decode("utf-8", errors="replace").splitlines():
        state = line.strip().split("|")[0].strip()
        if state:
            # sacct may return State+ suffix e.g. FAILED+OUT_OF_MEMORY
            return state.split("+")[0]
    return None


def _latest_slurm_log_excerpt(job_directory: str, job_id: str, max_chars: int = 4000) -> Optional[str]:
    if not job_id:
        return None
    log_path = os.path.join(job_directory, f"slurm-{job_id}.out")
    if not os.path.isfile(log_path):
        for name in os.listdir(job_directory):
            if name.startswith("slurm-") and name.endswith(".out"):
                log_path = os.path.join(job_directory, name)
                break
        else:
            return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, max_chars)
            f.seek(-read_size, os.SEEK_END)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def _load_status(job_directory: str) -> Optional[dict]:
    status_path = os.path.join(job_directory, "status.json")
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Error reading %s: %s", status_path, e)
        return None


def _save_status(job_directory: str, status: dict) -> None:
    status_path = os.path.join(job_directory, "status.json")
    status["last_updated"] = time.time()
    with open(status_path, "w") as f:
        json.dump(status, f)


def is_recent_training_activity(status: dict) -> bool:
    if status.get("status") != TrainingJobStatus.TRAINING:
        return False
    config = get_config()
    heartbeat_limit = config.get("training_heartbeat_seconds", 600)
    last = status.get("last_updated") or status.get("start_time", 0)
    return time.time() - last < heartbeat_limit


def job_should_block_resubmit(status: Optional[dict], job_name: str) -> bool:
    if is_slurm_job_active(job_name):
        return True
    if status and is_recent_training_activity(status):
        return True
    return False


def sync_training_job_statuses() -> None:
    """Update status.json from squeue/sacct for jobs with a SLURM job id."""
    config = get_config()
    training_dir = config["training_job_directory"]
    if not os.path.isdir(training_dir):
        return

    active_slurm = get_active_slurm_jobs()

    for entry in os.listdir(training_dir):
        job_directory = os.path.join(training_dir, entry)
        if not os.path.isdir(job_directory):
            continue

        status = _load_status(job_directory)
        if not status:
            continue

        current = status.get("status")
        if current in (
            TrainingJobStatus.COMPLETED,
            TrainingJobStatus.FAILED,
            "CANCELLED",
        ):
            continue

        job_name = entry
        job_id = status.get("job_id")

        if job_name in active_slurm:
            slurm_info = active_slurm[job_name]
            status["job_id"] = slurm_info["job_id"]
            status["slurm_state"] = slurm_info["state"]
            if current == TrainingJobStatus.QUEUED and slurm_info["state"] == "RUNNING":
                # Still QUEUED until training harness sets TRAINING.
                pass
            _save_status(job_directory, status)
            continue

        if not job_id:
            continue

        sacct_state = get_sacct_state(str(job_id))
        if not sacct_state:
            continue

        terminal = _SACCT_TERMINAL_STATUS.get(sacct_state)
        if not terminal:
            continue

        if current == TrainingJobStatus.TRAINING and terminal == TrainingJobStatus.COMPLETED:
            status["status"] = TrainingJobStatus.COMPLETED
        else:
            status["status"] = terminal

        status["slurm_state"] = sacct_state
        if terminal == TrainingJobStatus.FAILED:
            excerpt = _latest_slurm_log_excerpt(job_directory, str(job_id))
            if excerpt:
                status["slurm_log_tail"] = excerpt

        _save_status(job_directory, status)
        logger.info(
            "Synced job %s (slurm %s) -> %s (sacct %s)",
            job_name,
            job_id,
            status["status"],
            sacct_state,
        )
