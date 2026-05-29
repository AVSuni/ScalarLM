"""CLI entry point for distributed collective benchmarks."""

import argparse

from cray_infra.training.distributed import get_rank

from distributed_benchmarks import (
    run_collectives_benchmark,
    setup_distributed,
    teardown_distributed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cuda", "rocm", "cpu"], required=True)
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Reserved for future dtype support",
    )
    args = parser.parse_args()

    setup_distributed()
    try:
        results = run_collectives_benchmark(args.arch)
        if get_rank() == 0:
            print("\nBenchmark Results (GB/s):")
            for name, bw in results.items():
                print(f"{name}: {bw:.2e}")
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()

# torchrun --nnodes=1 --nproc-per-node=4 test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch cuda
# torchrun --nnodes=1 --nproc-per-node=4 test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch rocm
# torchrun --nnodes=1 --nproc-per-node=2 test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch cpu
