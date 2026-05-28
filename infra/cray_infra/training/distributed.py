import torch
import torch.distributed as dist


def _select_backend() -> str:
    # RCCL is exposed through the NCCL backend in PyTorch.
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init():
    if dist.is_initialized():
        return
    dist.init_process_group(backend=_select_backend())


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
    if dist.is_initialized():
        dist.barrier()


def _ensure_contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def allgather(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(sendbuf)
    recvbuf = _ensure_contiguous(recvbuf)
    dist.all_gather_into_tensor(recvbuf, sendbuf)
    return recvbuf


def reduce_scatter(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(sendbuf)
    recvbuf = _ensure_contiguous(recvbuf)
    dist.reduce_scatter_tensor(recvbuf, sendbuf, op=dist.ReduceOp.SUM)
    return recvbuf


def allreduce(tensor, op=dist.ReduceOp.SUM):
    _ensure_initialized()
    tensor = _ensure_contiguous(tensor)
    dist.all_reduce(tensor, op=op)
    return tensor


def alltoall(sendbuf, recvbuf=None):
    _ensure_initialized()
    sendbuf = _ensure_contiguous(sendbuf)
    world_size = get_size()
    if sendbuf.numel() % world_size != 0:
        raise ValueError("alltoall send buffer numel must be divisible by world size")

    input_list = list(sendbuf.chunk(world_size, dim=0))

    if recvbuf is None:
        recvbuf = torch.empty_like(sendbuf)
    recvbuf = _ensure_contiguous(recvbuf)
    output_list = list(recvbuf.chunk(world_size, dim=0))

    dist.all_to_all(output_list, input_list)
    return recvbuf


def send(tensor, dest: int):
    _ensure_initialized()
    tensor = _ensure_contiguous(tensor)
    dist.send(tensor, dst=dest)
    return tensor


def recv(tensor, source: int):
    _ensure_initialized()
    tensor = _ensure_contiguous(tensor)
    dist.recv(tensor, src=source)
    return tensor


def finalize():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
