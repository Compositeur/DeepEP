# MoE communication and compute overlap benchmark

This benchmark measures five scheduling windows around a SwiGLU MoE layer using
real DeepEP communication and explicit expert/shared-expert computation. It runs
on one four- or sixteen-rank Global Clos group. It does not run a training framework or predict a
complete model's training speedup.

## Workload and controls

| Window | Communication | Independent computation |
| --- | --- | --- |
| A | Forward dispatch | Shared-expert FC1, SwiGLU and FC2 |
| B | Forward combine | Output add consuming the combine result (dependency control) |
| R | Recompute dispatch | Eager attention replay on saved input (standalone proxy) |
| C | Backward cached dispatch | Shared-expert backward and expert forward recomputation |
| D | Backward combine of expert input gradients | Expert FC2 and FC1 weight gradients |

Each isolated window has `comm_only`, `compute_only`, `serial` and `overlap`
controls. B must wait before reading combine output even in the `overlap` control;
its raw timing differences are not evidence of legal overlap. R uses Q/K/V
projections, scaled QK matmul, causal FP32 softmax, value matmul and output projection.
It is not a full gated MLA/Flash4 implementation or a full checkpoint engine.

A complete layer defaults to `serial`, `a_only`, `d_only` and `ad`. Select
`--layer-suite ACD` for eight modes: `serial`, `a_only`, `c_only`, `d_only`, `ac`,
`ad`, `cd` and `acd`. These
modes use the same tensors, weights, routes and arithmetic; only the completion
wait moves across independent computation. `recompute_experts=true` adds the same expert forward
recomputation to every layer mode, using retained dispatched inputs, and feeds its
activations into dgrad. It does not replay the R communication or reproduce activation
offloading. C's isolated window always includes both shared backward and expert
recomputation; the default legacy layer configuration does not enable the latter.
DeepEP owns the communication stream;
expert and shared-expert operations run on the caller's compute stream.

A layer executes dispatch, shared and routed experts, forward combine, cached
backward dispatch, input-gradient computation, backward combine and deferred
expert weight gradients. The routed expert output enters forward combine; its
computed input gradient enters backward combine. Shared output/gradients are
included in the results. Routes are fixed per run, drawn without replacement per
token, with equal FP32 gate weights. This is not a learned router or a captured
particular production token distribution.

## Requirements

- Python 3.11+, Torch and matching `torch_npu`, CANN and driver versions compatible
  with the installed DeepEP build. Follow the repository [build guide](../../docs/build.md).
- One allocated Global Clos EP4 or EP16 group, launched with `WORLD_SIZE=ep`.
  Each node's visible NPU count must equal `LOCAL_WORLD_SIZE`. EP16 may require
  a multi-node launch; use the repository's network/endpoint setup for that topology.
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

Use `--case A`, `B`, `R`, `C` or `D` for one isolated window, `--case layer` for
layer modes without isolated windows, or the default `--case all` for all five
windows plus the selected layer suite.
Correctness preparation still executes the layer for window-only runs so that
B/C/D use real operands and handles. Isolated serial/overlap outputs are also checked;
A/R use an extra identity roundtrip only outside timing, avoiding comparisons of
unstable receive row order. Attention proxy weights are allocated before timing. Fixtures and cached handles
are refreshed outside timing between cases, after fresh-dispatch windows.

`configs/smoke_v1.json` and `smoke_v2.json` have small inputs and two repetitions
with two samples each. They test execution, not stable performance.
`configs/example_v1.json` supplies a larger illustrative workload.
`configs/example_ep16_v2.json` enables EP16, 8192 tokens per rank, H=2816, E=256,
K=8, expert/shared width 768, clamp 10, expert recomputation and the ACD suite.
This defines reference workload dimensions. The preset does not model a full PP
schedule, allocation/offloading, MLA, detailed routing load or per-kernel 32/48/64
communication controls. It is a candidate to validate, not a calibrated reproduction.
`configs/example_ep16_v2_std006.json` keeps the same dimensions and scheduling,
uses `weight_init_std=0.006`, FP32 reference checking and five repetitions of 30
samples. It is a separate numerical workload: it cannot qualify or replace the
original scale-0.1 configuration. The scale-0.1 large workload has failed the
sampled reference tolerance during device validation; retain and investigate that
failure rather than describing it as accepted.
A preset alone is not proof of hardware support or measured benefit. Increase warmup, iterations and
repeats in a configuration copy after verifying correctness and memory requirements.
For V2, retain legacy `num_sms=0` and set independent `dispatch_num_sms`,
`combine_num_sms` and `reduce_num_sms`. Zero means the backend default (dispatch)
or inheritance from the dispatch handle (combine/reduce). Explicit controls require
V2/UDMA; V1 rejects them. For example, dispatch/combine=16 with reduce=32 keeps
the reduction launch count unchanged while lowering communication launch counts.
Cached dispatch inherits its handle's dispatch count. Reports record effective
counts from the handle; verify native kernel blocks in separate device traces.
These are communication AIV launch controls, not compute-core reservations.
For a 16/16/32 experiment, copy a workload configuration and set:

```json
{
  "api": "v2_elastic",
  "num_sms": 0,
  "dispatch_num_sms": 16,
  "combine_num_sms": 16,
  "reduce_num_sms": 32
}
```

This fragment must be combined with the original workload fields. Do not replace
an EP16 configuration with this fragment alone, because omitted fields use defaults.
Changing counts changes the configuration signature. Establish same-count serial
and overlap measurements, then compare absolute layer times across configurations;
do not label a cross-core comparison as the same-mode code regression check.

## Correctness and timing

Device execution first checks tiny independent float64/autograd mathematics,
then sampled source-token Y/dX and local expert dW slices, followed by all six
outputs across layer modes: Y, dX, expert dW1/dW2 and shared-expert dW1/dW2.
Source samples default to FP32 autograd with BF16-valued inputs/weights. Set
`source_reference_dtype` to `"bf16"` explicitly to check against independent CPU
autograd with BF16 operator boundaries, including BF16 communication reduction
outputs before the shared branch is added. Both modes use rtol 0.06, atol 0.002;
the BF16 mode also records FP32 error diagnostics without claiming FP32 acceptance.
The selected dtype is part of the workload signature, so baselines from different
reference modes are not interchangeable. BF16 boundary rounding may differ from
an all-FP32 graph, especially for near-zero cancellation. No tolerance is changed
automatically. The SiLU derivative uses the backward of the actual SiLU operator
to avoid additional BF16 rounding from an expanded sigmoid expression.
`weight_init_std` specifies the deterministic expert/shared-expert weight scale
(default 0.1). Input/gradient generation keeps its original scale. Changing this
field changes the numerical workload and invalidates baseline comparisons; a run
with another scale does not qualify a failed original configuration. Both the
selected reference and FP32 diagnostics record sample errors before acceptance.
Sampled dW uses rtol 0.04, atol 0.002. Cross-mode defaults are
rtol 0.02, atol 0.002. Sampled reference checks do not prove every distributed
route against an independent oracle. Failed correctness prevents performance
aggregation.

NPU events measure A/C/D windows and the layer from forward dispatch through the
final gradient join. Synchronization, rank barriers and gradient zeroing precede
each timed sample. Layer modes and isolated controls rotate order between repetitions. Reports take the slowest rank for each sample,
then compute independent repetition means. Benefit is the serial-to-overlap
percentage reduction with a conservative 95% t interval. `inconclusive` is a valid
result; the benchmark does not require or manufacture positive speedup. Small
sample counts, thermal drift or device contention can invalidate conclusions.

Set `profile_iterations` to a positive value for separate serial/overlap traces
collected after all performance samples. `--case all` records layer serial/combined
modes and serial/overlap for each isolated window under separate `profile_CASE_MODE`
directories. Trace-enabled time is not used for
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
  tests/test_moe_overlap_cli.py tests/test_moe_overlap_windows.py \
  tests/test_moe_overlap_trace.py -q
python -m benchmarks.moe_overlap --cpu-check --out cpu-math
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torch.distributed.run \
  --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29572 \
  --nproc-per-node=4 --module benchmarks.moe_overlap.cpu_reference \
  --out benchmark-runs/moe-overlap/cpu-ep4
```

The four CPU/Gloo processes check EP partitioning and gradients against autograd,
including duplicate routing, imbalance and an empty expert. Add `--layer-suite ACD
--recompute-experts` to that CPU launch to check all eight modes with recomputation. They do not call
DeepEP or measure NPU performance. CPU reports explicitly disable performance claims.
The framework does not cover router parameter gradients, full MLA attention, complete checkpoint
recomputation, pipeline/data parallel scheduling, optimizer steps, EP8, LL or
complete-model acceptance. R is isolated, and expert recomputation uses retained inputs;
full checkpoint dispatch and offload scheduling remain outside layer acceptance. The optional pointer backend and multi-NPU paths need
separate device validation; offline tests cannot establish speedup.

The backward scheduling arrangement was informed by the
[pointer-gmm EP4 example](https://gitcode.com/lishaoxun/ascend_deepep/tree/pointer-gmm-ep4-repro/benchmarks/pointer_gmm_ep4)
at reference commit `b2ed69b11e3a994cb13dab9aee9ec8e4593e366f`.
The public V1 integration also follows
[MegatronAdaptor PR 217](https://gitcode.com/Ascend/MegatronAdaptor/pull/217).
Those references do not establish measured training gains.

## Reference overlap values

`configs/reported_overlap.json` provides the following reference values, not acceptance
thresholds. Their denominator is **communication duration**: intersect
communication and computation within the same fused-layer invocation, then sum
intersection and communication durations over selected calls/ranks before dividing.
Full communication is counted; payload-only ratios are a different metric.

| Case | Reference temporal overlap |
| --- | ---: |
| A | 18.35% |
| B | 0.00% |
| R | 80.00% |
| C | 99.69% |
| D | 82.19% |

The C reference combines shared-expert backward (66.987%) and expert forward
recomputation (32.706%), totaling 99.693% before rounding. A shared-backward-only
workload is a different reference case. R covers attention recomputation during backward,
not attention backward. The reference configuration is EP16, MBS=2, GBS=256, PP=2,
TP=CP=1, sequence length 4096; the reference records these dimensions. Equal gates,
random routes and eager attention define a standalone approximation.

```bash
python -m benchmarks.moe_overlap --layer-suite ACD \
  --reference-overlap benchmarks/moe_overlap/configs/reported_overlap.json
```

A reference annotates reports; it never changes computation size, inserts sleeps,
sets measured timings or demands a target speedup. EP4 measurements are marked as
a workload mismatch against the EP16 reference configuration. Percentages alone provide no
kernel durations, critical path or training-step savings. A high overlap ratio may
coexist with slower communication/compute under contention. Measure window/layer
latency separately and re-establish baselines after this benchmark revision.

## Measure device temporal overlap

Use a reviewed device PID/category and kernel selectors for one rank, one layer
invocation and one narrow window. Names alone do not establish phase attribution.
Ascend trace device categories may be absent (treated as the empty string); never select host ranges
as device work. Inspect the actual trace before replacing these placeholders:

```bash
python -m benchmarks.moe_overlap.trace --trace /path/to/trace_view.json \
  --case C --rank 0 --device-pid DEVICE_PID --category '' \
  --start-us WINDOW_START --end-us WINDOW_END \
  --comm 'REVIEWED_COMM_KERNEL_REGEX' --compute 'REVIEWED_COMPUTE_KERNEL_REGEX' \
  --reference-overlap benchmarks/moe_overlap/configs/reported_overlap.json \
  --out trace-c-rank0-call1
```

The incremental reader accepts Chrome event arrays or `traceEvents` objects.
Within each selected device process it clips intervals, unions repeated/overlapping
kernels and intersects the communication and computation unions. Ambiguous
selectors, missing work and invalid intervals fail. Reports separately provide
communication coverage, computation coverage, overlap/window time and exposed
communication microseconds. `performance_benefit` is null: temporal intersection
is not a measurement of speedup. Neither trace analysis nor replay automatically
classifies original training phases or guarantees matching workload conditions.

Aggregate reviewed per-call reports of the **same case and kernel selectors**:

```bash
python -m benchmarks.moe_overlap.trace --aggregate \
  benchmark-runs/moe-overlap/trace-c-rank0-call1/temporal_overlap.json \
  benchmark-runs/moe-overlap/trace-c-rank8-call1/temporal_overlap.json \
  --reference-overlap benchmarks/moe_overlap/configs/reported_overlap.json \
  --out trace-c-weighted
```

This computes `sum(overlap_us) / sum(communication_us)`, not the mean of rank
percentages, and rejects duplicate rank/window reports. It never intersects clocks
from different ranks; summed durations are accounting totals, not elapsed step
time. A numeric difference from the supplied reference is a diagnostic, not an
acceptance or speedup verdict. If a different reference's denominator is unknown,
set `ratio_definition` to `unspecified`; no numerical comparison is made.
