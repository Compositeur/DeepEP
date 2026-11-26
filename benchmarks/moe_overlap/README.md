# MoE communication and compute overlap benchmark

This benchmark compares two overlap points in one SwiGLU MoE layer using real
DeepEP communication and explicit expert/shared-expert computation. It runs on one
four-rank Global Clos group. It does not run a training framework or predict a
complete model's training speedup.

## Workload and controls

| Window | Communication | Independent computation |
| --- | --- | --- |
| A | Forward dispatch | Shared-expert FC1, SwiGLU and FC2 |
| D | Backward combine of expert input gradients | Expert FC2 and FC1 weight gradients |

Each isolated window has `comm_only`, `compute_only`, `serial` and `overlap`
controls. A complete layer has `serial`, `a_only`, `d_only` and `ad` modes. These
modes use the same tensors, weights, routes and arithmetic; only the completion
wait moves across independent computation. DeepEP owns the communication stream;
expert and shared-expert operations run on the caller's compute stream.

A layer executes dispatch, shared and routed experts, forward combine, cached
backward dispatch, input-gradient computation, backward combine and deferred
expert weight gradients. The routed expert output enters forward combine; its
computed input gradient enters backward combine. Shared output/gradients are
included in the results. Routes are fixed per run, drawn without replacement per
token, with equal FP32 gate weights. This is not a learned router or a captured
recorded production token distribution.

## Requirements

- Python 3.11+, Torch and matching `torch_npu`, CANN and driver versions compatible
  with the installed DeepEP build. Follow the repository [build guide](../../docs/build.md).
- Exactly four visible, allocated NPUs in the same supported Global Clos EP group.
  The launcher never finds free devices or terminates other jobs.
- V1 public `Buffer(machine_type="global_clos")` or V2 public `ElasticBuffer`.
  They use separate configurations and baselines. V2 uses expanded HT layout and
  capacity of at least 257 tokens even for the small smoke workload.
- Default `gmm_add` uses `npu_grouped_matmul` and FP32 gradient accumulation.
  Optional `pointer` requires an independently installed `pointer_gmm` module
  providing `npu_grouped_matmul_add_ptr_`; missing support is an explicit error.

The reference revision is `83252a54d2363734a706a7dd8368bebc3094af5a`.
It documents the API used during development, not a forced runtime version.
Reports record actual import paths, binary hashes and any discoverable Git revision.
An import or version string alone does not establish which source built a wheel.

## Plan and software checks

Run commands from the repository root. The default only prints a plan, needs no
Torch, and creates no result files:

```bash
python -m benchmarks.moe_overlap
python -m benchmarks.moe_overlap --case D
```

Use an isolated environment for execution. Load the matching CANN environment
before importing `torch_npu`:

```bash
source /path/to/ascend-toolkit/set_env.sh
python -m benchmarks.moe_overlap --check-env --out environment-v1
python -m benchmarks.moe_overlap --check-env \
  --config benchmarks/moe_overlap/configs/smoke_v2.json --out environment-v2
```

`--check-env` checks imports, GMM schema and required method parameters. It does
not create a Buffer or allocate NPU tensors. A passing report is not native
communication, numerical, stream-safety or hardware validation.

`--workspace` selects result and cache storage; its default is
`benchmark-runs/moe-overlap/`. `--out` is a new directory relative to that workspace,
or an absolute directory inside it. Existing output and path escapes are rejected.
No package installation is performed by the benchmark.

## Four-rank device execution

The following device list and endpoint are examples. Replace them with allocated
resources and the endpoint required by your installed DeepEP backend; the
benchmark does not reserve cards or network ports.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
python -m torch.distributed.run --nnodes=1 --node-rank=0 \
  --master-addr=127.0.0.1 --master-port=29571 --nproc-per-node=4 \
  --module benchmarks.moe_overlap --execute --out smoke-v1
```

Use `--case A` or `--case D` for one isolated window, `--case layer` for the four
layer modes without isolated windows, or the default `--case all` for both.
Correctness preparation still executes the layer for window-only runs so that D
uses operands and handles produced by real dispatch and input-gradient computation.

`configs/smoke_v1.json` and `smoke_v2.json` have small inputs and two repetitions
with two samples each. They test execution, not stable performance.
`configs/example_v1.json` supplies a larger illustrative workload. Neither preset
is proof of hardware support or measured benefit. Increase warmup, iterations and
repeats in a configuration copy after verifying correctness and memory requirements.
No communication-kernel count control is claimed for V2 (`num_sms=0`).

## Correctness and timing

Device execution first checks tiny independent float64/autograd mathematics,
then sampled source-token Y/dX and local expert dW slices, followed by all six
outputs across layer modes: Y, dX, expert dW1/dW2 and shared-expert dW1/dW2.
Source samples use FP32 autograd with BF16-valued inputs/weights (rtol 0.06,
atol 0.002); sampled dW uses rtol 0.04, atol 0.002. Cross-mode defaults are
rtol 0.02, atol 0.002. Sampled reference checks do not prove every distributed
route against an independent oracle. Failed correctness prevents performance
aggregation.

NPU events measure A/D windows and the layer from forward dispatch through the
final gradient join. Synchronization, rank barriers and gradient zeroing precede
each timed sample. Layer modes rotate order between repetitions; isolated controls
currently run in a fixed order. Reports take the slowest rank for each sample,
then compute independent repetition means. Benefit is the serial-to-overlap
percentage reduction with a conservative 95% t interval. `inconclusive` is a valid
result; the benchmark does not require or manufacture positive speedup. Small
sample counts, thermal drift or device contention can invalidate conclusions.

Set `profile_iterations` to a positive value for separate serial/overlap traces
collected after all performance samples. Trace-enabled time is not used for
performance comparisons. Actual simultaneous kernel execution must be established
from device traces, not from moving a wait in Python.

## Baseline comparison

Collect a full-layer report under a fixed configuration and software/hardware
environment, repeat the same commit to measure natural variation, and then test
a changed DeepEP build with the same benchmark code:

```bash
python -m benchmarks.moe_overlap --compare \
  benchmark-runs/moe-overlap/baseline/summary.json \
  benchmark-runs/moe-overlap/candidate/summary.json \
  --margin-percent 1 --out comparison
```

The example 1% margin is user-chosen, not a calibrated acceptance threshold.
Comparison checks workload, framework fingerprint and recorded environment.
DeepEP artifacts may differ because they are the subject of the comparison.
It rejects CPU results and window-only reports as layer baselines. Exit code 2
means invalid input/comparison; 3 means a statistically detected regression;
0 may include inconclusive results. Device execution can fail with other nonzero
codes; none imply that a performance baseline was established.

## CPU development checks and limitations

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/test_moe_overlap.py tests/test_moe_overlap_environment.py \
  tests/test_moe_overlap_cli.py -q
python -m benchmarks.moe_overlap --cpu-check --out cpu-math
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torch.distributed.run \
  --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29572 \
  --nproc-per-node=4 --module benchmarks.moe_overlap.cpu_reference \
  --out benchmark-runs/moe-overlap/cpu-ep4
```

The four CPU/Gloo processes check EP partitioning and gradients against autograd,
including duplicate routing, imbalance and an empty expert. They do not call
DeepEP or measure NPU performance. CPU reports explicitly disable performance claims.
The framework does not cover router parameter gradients, attention, checkpoint
recomputation, pipeline/data parallel scheduling, optimizer steps, EP8, LL or
complete-model acceptance. The optional pointer backend and four-NPU paths need
separate device validation; offline tests cannot establish speedup.

The backward scheduling arrangement was informed by the
[pointer-gmm EP4 example](https://gitcode.com/lishaoxun/ascend_deepep/tree/pointer-gmm-ep4-repro/benchmarks/pointer_gmm_ep4)
at reference commit `b2ed69b11e3a994cb13dab9aee9ec8e4593e366f`.
The public V1 integration also follows
[MegatronAdaptor PR 217](https://gitcode.com/Ascend/MegatronAdaptor/pull/217).
Those references do not establish measured training gains.
