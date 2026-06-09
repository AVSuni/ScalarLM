import os
import sys
import time

import torch
import torch.distributed as dist

from cray_infra.training.train_debug import is_train_debug_enabled


def _trace(msg: str) -> None:
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    line = f"[rank={rank}] dist [{time.monotonic():.3f}]: {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()


def _cuda_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device("cuda", local_rank)


def cuda_device() -> torch.device:
    return _cuda_device()


def _to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_cuda:
        return tensor
    return tensor.to(_cuda_device())


def init():
    _trace(
        f"init() entered pid={os.getpid()} "
        f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}"
    )
    if dist.is_initialized():
        _trace("init() skipped, already initialized")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for distributed training")

    _trace("calling init_process_group(backend=nccl)")
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _trace(
        f"set_device({local_rank}) after init "
        f"cuda_current_device={torch.cuda.current_device()}"
    )
    _trace(
        f"init_process_group complete rank={dist.get_rank()} "
        f"world_size={dist.get_world_size()} backend={dist.get_backend()}"
    )


def _ensure_initialized():
    if not dist.is_initialized():
        init()


def get_rank():
    _ensure_initialized()
    return dist.get_rank()


def get_size():
    _ensure_initialized()
    return dist.get_world_size()


def barrier():
    if not dist.is_initialized():
        return
    dist.barrier()


def _ensure_contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def allgather(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(_to_cuda(sendbuf))
    recvbuf = _ensure_contiguous(_to_cuda(recvbuf))
    dist.all_gather_into_tensor(recvbuf, sendbuf)
    return recvbuf


def reduce_scatter(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(_to_cuda(sendbuf))
    recvbuf = _ensure_contiguous(_to_cuda(recvbuf))
    dist.reduce_scatter_tensor(recvbuf, sendbuf, op=dist.ReduceOp.SUM)
    return recvbuf


def allreduce(tensor, op=dist.ReduceOp.SUM):
    _ensure_initialized()
    tensor = _ensure_contiguous(_to_cuda(tensor))
    dist.all_reduce(tensor, op=op)
    return tensor


def alltoall(sendbuf, recvbuf=None):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(_to_cuda(sendbuf))
    world_size = get_size()
    if sendbuf.numel() % world_size != 0:
        raise ValueError("alltoall send buffer numel must be divisible by world size")

    input_list = [_ensure_contiguous(chunk) for chunk in sendbuf.chunk(world_size, dim=0)]

    if recvbuf is None:
        recvbuf = torch.empty_like(sendbuf)
    recvbuf = _ensure_contiguous(_to_cuda(recvbuf))
    output_list = list(recvbuf.chunk(world_size, dim=0))

    dist.all_to_all(output_list, input_list)
    return recvbuf


def send(tensor, dest: int):
    _ensure_initialized()
    tensor = _ensure_contiguous(_to_cuda(tensor))
    dist.send(tensor, dst=dest)
    return tensor


def recv(tensor, source: int):
    _ensure_initialized()
    tensor = _ensure_contiguous(_to_cuda(tensor))
    dist.recv(tensor, src=source)
    return tensor


def finalize():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
