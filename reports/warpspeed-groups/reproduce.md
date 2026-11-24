# Reproduce the WarpSpeed measurements

Use a checkout containing the candidate, CUDA 13.5, GCC 14 and CMake 3.30.4 or newer. The harness builds both revisions against one instance of NVBench. The NVBench commit supplies GPU clock measurements and the entropy stopping criterion.

```bash
export CCCL_REPO="$(git rev-parse --show-toplevel)"
export CUDA_PATH=/usr/local/cuda-13.5
export CXX=/usr/bin/g++-14
export CUDA_VISIBLE_DEVICES=0  # Select the RTX PRO 6000 on this machine.
work_dir="$(mktemp -d)"
tar xf "$CCCL_REPO/reports/warpspeed-groups/harness.tar.gz" -C "$work_dir"
cd "$work_dir/harness"
mkdir baseline
git -C "$CCCL_REPO" archive 4db67c68d9f3177a06b403002af619cc420fb200 cub/cub cudax/include libcudacxx/include thrust/thrust | tar x -C baseline
export NVBENCH_SOURCE="$work_dir/nvbench"
export NVBENCH_BUILD="$work_dir/nvbench-build"
git clone https://github.com/NVIDIA/nvbench.git "$NVBENCH_SOURCE"
git -C "$NVBENCH_SOURCE" checkout 2514944a603a71f37cde6aeee7384fda94959542
cmake -S . -B "$NVBENCH_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
cmake --build "$NVBENCH_BUILD" --target nvbench.main
python3 build.py old "$PWD/baseline"
python3 build.py final "$CCCL_REPO"
python3 build.py oldgraph "$PWD/baseline"
python3 build.py finalgraph "$CCCL_REPO"
python3 pairs.py final final 8
python3 summarize.py final final
WARP_BASELINE=oldgraph WARP_RESOURCE_PROBE=1 python3 pairs.py finalgraph graph 8 -a 'N[pow2]=[16,20]' --timeout 0.35
WARP_BASELINE=oldgraph python3 summarize.py finalgraph graph
export WARP_GRAPH_REPEATS=192
python3 build.py longgraph "$PWD/baseline"
python3 build.py longcandidategraph "$CCCL_REPO"
WARP_BASELINE=longgraph python3 pairs.py longcandidategraph graph192 10 -b scan_u64 -a 'N[pow2]=16' -b rle_floor -a 'N[pow2]=20' -a Run=1 --timeout 0.45
WARP_BASELINE=longgraph python3 summarize.py longcandidategraph graph192
python3 code.py old final
```

The suite takes a cold measurement of one operation per sample. The graph suite captures 24 calls in one graph. Its summarizer divides elapsed time by 24. Each comparison alternates the order of the two executables on every pair. Both suites run sample correctness checks on the GPU before timing each case. Disassembly, compiler resource logs, summary CSVs and all NVBench JSON and Markdown output stay in the harness directory.

The distributed harness supplies a valid lookback policy for Clang to instantiate. The timed path selects the lookahead policy on this GPU. We verified that the distributed harness reproduces all eight baseline and final scan/RLE device kernels instruction-for-instruction.

The harness passes all four checkout include directories to NVCC (`cub`, `cudax/include`, `libcudacxx/include`, and `thrust`). CUB reads the experimental group implementation directly from `cudax/include`.
