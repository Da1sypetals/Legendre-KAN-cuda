# CUDA implementation of Legendre KAN

## References:

- [pytorch implementation](https://github.com/1ssb/torchkan)

## Note

- There are no optimizations other than memory access coalescing. I a cuda beginner and willing to receive optimization suggestions : )

## Start

1. Install

```bash
pip install -e .
```

> Make sure the version of nvcc in PATH is compatible with your current PyTorch version (it seems minor version difference is OK).

2. Run

   - Run test on MNIST:

   ```bash
   python cheby_test.py
   ```

   - Run benchmark (code from [KAN-benchmarking](https://github.com/Jerry-Master/KAN-benchmarking)):

   ```bash
   python benchmark.py --method all --reps 100 --just-cuda
   ```
