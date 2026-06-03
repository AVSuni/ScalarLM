from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.print_logo import print_logo

from cray_megatron.megatron.training_loop import TrainingLoop, get_max_steps
from cray_megatron.megatron.training_harness import TrainingHarness

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def _trace_trainer(msg: str) -> None:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    line = f"[rank={rank}] trainer [{time.monotonic():.3f}]: {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()


class MegatronTrainer:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

    def train(self):
        self.train_loop()

    def train_loop(self):
        _trace_trainer("train_loop enter")
        _trace_trainer("train_loop calling update_status(TRAINING)")
        self.training_harness.update_status(
            status=TrainingJobStatus.TRAINING, metadata={"max_steps": get_max_steps()}
        )
        _trace_trainer("train_loop update_status(TRAINING) complete")

        _trace_trainer("train_loop calling print_logo")
        print_logo()
        _trace_trainer("train_loop print_logo complete")

        _trace_trainer("train_loop creating TrainingLoop")
        training_loop = TrainingLoop(self.training_harness)
        _trace_trainer("train_loop calling TrainingLoop.train()")
        training_loop.train()
        _trace_trainer("train_loop TrainingLoop.train() complete")

        _trace_trainer("train_loop calling update_status(COMPLETED)")
        self.training_harness.update_status(status=TrainingJobStatus.COMPLETED)
        _trace_trainer("train_loop complete")
