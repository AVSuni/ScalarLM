import faulthandler
import logging
import os
import signal
import sys
import time
import traceback

from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.huggingface.get_hf_token import get_hf_token
from cray_megatron.megatron.training_harness import TrainingHarness
from cray_infra.training.distributed import init, finalize

logger = logging.getLogger(__name__)

faulthandler.enable()
faulthandler.dump_traceback_later(timeout=600, repeat=True)


def _boot(msg: str) -> None:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    line = f"[rank={rank}] boot pid={os.getpid()} t={time.monotonic():.3f}: {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()


_boot("main.py entered")


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def main():
    _boot("calling init()")
    init()
    _boot("init() complete")

    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _boot(
        f"set_device({local_rank}) after init "
        f"cuda_current_device={torch.cuda.current_device()}"
    )

    _boot("reconfigure stdout")
    sys.stdout.reconfigure(line_buffering=True)

    _boot("create TrainingHarness")
    harness = TrainingHarness()

    _boot("get_hf_token")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()
    _boot("get_hf_token complete")

    try:
        _boot("setup_logging")
        setup_logging()
        _boot("setup_signal_handler")
        setup_signal_handler(harness)

        _boot("import megatron_trainer")
        from cray_megatron.megatron.megatron_trainer import MegatronTrainer

        _boot("megatron_trainer imported")

        _boot("create MegatronTrainer")
        trainer = MegatronTrainer(training_harness=harness)
        _boot("MegatronTrainer created")

        _boot("calling trainer.train()")
        trainer.train()
        _boot("trainer.train() complete")
    except Exception as e:
        _boot(f"training failed: {e}")
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        raise e

    _boot("calling finalize()")
    finalize()
    _boot("finalize() complete")


def setup_logging():
    logging.basicConfig(level=logging.INFO)

    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("cray_megatron.megatron.distribution.fsdp").setLevel(
        logging.INFO
    )


def setup_signal_handler(harness):
    def terminate_handler(sig, frame):
        logger.warning("Received termination signal %s", sig)
        harness.update_status(
            status=TrainingJobStatus.FAILED,
            metadata={"error": f"Terminated by signal {sig}"},
        )
        sys.exit(1)

    signal.signal(signal.SIGTERM, terminate_handler)
    signal.signal(signal.SIGINT, terminate_handler)


main()
