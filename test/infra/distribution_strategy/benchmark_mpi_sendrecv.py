"""CLI entry point for distributed send/recv benchmarks."""

import argparse

from cray_infra.training.distributed import get_rank, get_size

from distributed_benchmarks import (
    run_sendrecv_benchmark,
    setup_distributed,
    teardown_distributed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cuda", "rocm", "cpu"])
    parser.add_argument(
        "--arch_type",
        choices=["cuda", "rocm", "cpu"],
        help="Deprecated alias for --arch",
    )
    args = parser.parse_args()
    arch = args.arch or args.arch_type
    if arch is None:
        parser.error("one of --arch or --arch_type is required")

    setup_distributed()
    try:
        assert get_size() == 2, "This benchmark only works with two ranks."
        print(f"Rank {get_rank()} of {get_size()} running with arch={arch}")
        bandwidth = run_sendrecv_benchmark(arch)
        if get_rank() == 0:
            print(f"Send/Recv (GB/s): {bandwidth:.6f}")
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
