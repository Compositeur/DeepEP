"""Independent A/C/D waits and five isolated communication/compute windows."""

import torch
from .model import (
    Math,
    initialize,
    expert_forward,
    expert_dgrad,
    expert_wgrad,
    shared_forward,
    shared_backward,
    attention_replay,
)
from .scenarios import WINDOWS, CONTROLS, modes_for_suite


class Clock:
    def __init__(self):
        self.events = {}

    def mark(self, name):
        event = torch.npu.Event(enable_timing=True)
        event.record()
        self.events[name] = event

    def results(self, pairs):
        self.events["end"].synchronize()
        return {key: self.events[start].elapsed_time(self.events[end]) * 1000 for key, start, end in pairs}


class Layer:
    def __init__(self, c, adapter, rank=0, npu=False):
        self.c, self.adapter, self.npu = c, adapter, npu
        self.math = Math(npu, c.wgrad_backend)
        self.data = initialize(
            c, rank, "npu" if npu else "cpu", torch.bfloat16 if npu else torch.float64, all_experts=not npu
        )
        dtype = torch.float32 if npu else torch.float64
        self.g1 = torch.zeros_like(self.data["w1"], dtype=dtype)
        self.g2 = torch.zeros_like(self.data["w2"], dtype=dtype)
        self.last_d = None
        self.last_f = None
        self.last_c = None
        self.attention_fixture = None

    def reset(self):
        self.g1.zero_()
        self.g2.zero_()

    def step(self, overlap_a=False, overlap_d=False, timed=False, *, overlap_c=False):
        p = self.data
        c = self.c
        adapter = self.adapter
        clock = Clock() if timed else None

        def mark(name):
            if clock:
                clock.mark(name)

        mark("start")
        mark("a_start")
        with torch.profiler.record_function("moe/A/dispatch_shared"):
            rx, ctx, done = adapter.dispatch(p["x"], p["routes"], p["gates"])
            if not overlap_a:
                adapter.wait(done)
            ys, sctx = shared_forward(p["x"], p["sw1"], p["sw2"], c.clamp)
            if overlap_a:
                adapter.wait(done)
        mark("a_end")
        x, gates, counts = adapter.unpack(rx, ctx)
        with torch.profiler.record_function("moe/experts_forward"):
            z, ectx = expert_forward(self.math, x, p["w1"], p["w2"], counts, c.clamp)
            weighted = (z.float() * gates[:, None]).to(z.dtype) if self.npu else z * gates[:, None]
            payload = adapter.prepare_combine(weighted, ctx)
        routed, done = adapter.combine(payload, ctx)
        adapter.wait(done)
        y = routed + ys
        self.last_f = (payload, ctx, ys, routed)
        mark("c_start")
        with torch.profiler.record_function("moe/backward_cached_dispatch"):
            dy, done = adapter.cached(p["dy"], ctx)
            if not overlap_c:
                adapter.wait(done)
            dxshared, sg1, sg2 = shared_backward(p["dy"], p["sw1"], p["sw2"], sctx)
            if c.recompute_experts:
                _, ectx = expert_forward(self.math, x, p["w1"], p["w2"], counts, c.clamp)
            if overlap_c:
                adapter.wait(done)
        mark("c_end")
        self.last_c = (ctx, sctx, x, counts)
        with torch.profiler.record_function("moe/pack_cached_gradient"):
            dy = adapter.pack_gradient(dy, ctx)
        with torch.profiler.record_function("moe/experts_dgrad"):
            dxpacked, wctx = expert_dgrad(self.math, dy, gates, p["w1t"], p["w2t"], counts, ectx)
            payload = adapter.prepare_combine(dxpacked, ctx)
        # Capture only after aggregate/casts have produced the final communication input.
        mark("d_start")
        with torch.profiler.record_function("moe/D/combine_wgrad"):
            dx, done = adapter.combine(payload, ctx)
            if not overlap_d:
                adapter.wait(done)
            expert_wgrad(self.math, wctx, counts, self.g1, self.g2)
            if overlap_d:
                adapter.wait(done)
        mark("d_end")
        dx = dx + dxshared
        mark("end")
        self.last_d = (payload, ctx, wctx, counts)
        outputs = dict(Y=y, dX=dx, expert_dW1=self.g1, expert_dW2=self.g2, shared_dW1=sg1, shared_dW2=sg2)
        times = (
            clock.results(
                [
                    ("total_us", "start", "end"),
                    ("a_window_us", "a_start", "a_end"),
                    ("c_window_us", "c_start", "c_end"),
                    ("d_window_us", "d_start", "d_end"),
                ]
            )
            if clock
            else {}
        )
        return outputs, times

    def prepare_windows(self, cases):
        """Allocate attention proxy weights outside timed windows, only when needed."""
        if "R" not in cases or self.attention_fixture is not None:
            return
        c = self.c
        x = self.data["x"][: min(c.tokens, c.attention_tokens)]
        generator = torch.Generator().manual_seed(c.seed + 50000)
        width = c.attention_heads * c.attention_head_dim

        def rand(shape):
            return (torch.randn(shape, generator=generator) * 0.1).to(device=x.device, dtype=x.dtype)

        self.attention_fixture = (x, rand((c.hidden, 3 * width)), rand((width, c.hidden)), c.attention_heads)

    def window(self, case, control, *, timed=True):
        """B is dependency constrained; R is an isolated, explicitly synthetic proxy."""
        if case not in WINDOWS or control not in CONTROLS:
            raise ValueError("Unknown window/control")
        if case == "R" and self.attention_fixture is None:
            raise RuntimeError("Prepare R attention fixture before timing")
        if case in ("B", "C", "D") and self.last_d is None:
            raise RuntimeError("Prepare window fixtures with a complete step first")
        p = self.data
        adapter = self.adapter
        clock = Clock() if timed else None
        if clock:
            clock.mark("start")
        comm = control != "compute_only"
        compute = control != "comm_only"
        done = None
        communication_output = compute_output = None
        if case in ("A", "R"):
            if comm:
                communication_output, context, done = adapter.dispatch(p["x"], p["routes"], p["gates"])
            if control == "serial":
                adapter.wait(done)
            if compute:
                if case == "A":
                    compute_output = shared_forward(p["x"], p["sw1"], p["sw2"], self.c.clamp)[0]
                else:
                    compute_output = attention_replay(*self.attention_fixture)
        elif case == "B":
            payload, ctx, ys, routed = self.last_f
            if comm:
                communication_output, done = adapter.combine(payload, ctx)
            # The representative output-add consumes combine's result. Moving this
            # wait would read an unfinished tensor, not create legal overlap.
            if comm and (compute or control == "serial"):
                adapter.wait(done)
                done = None
            if compute:
                compute_output = (communication_output if comm else routed) + ys
        elif case == "C":
            ctx, sctx, expert_x, counts = self.last_c
            if comm:
                communication_output, done = adapter.cached(p["dy"], ctx)
            if control == "serial":
                adapter.wait(done)
            if compute:
                shared_grads = shared_backward(p["dy"], p["sw1"], p["sw2"], sctx)
                recomputed, _ = expert_forward(self.math, expert_x, p["w1"], p["w2"], counts, self.c.clamp)
                compute_output = (*shared_grads, recomputed)
        elif case == "D":
            payload, ctx, wctx, counts = self.last_d
            if comm:
                communication_output, done = adapter.combine(payload, ctx)
            if control == "serial":
                adapter.wait(done)
            if compute:
                expert_wgrad(self.math, wctx, counts, self.g1, self.g2)
                compute_output = (self.g1, self.g2)
        if done is not None and control != "serial":
            adapter.wait(done)
        if clock:
            clock.mark("end")
            return clock.results([("window_us", "start", "end")])["window_us"]
        if comm and case in ("A", "R"):
            # Fresh receive order is not a stable oracle. Validate an identity
            # roundtrip outside timing instead of comparing transport row order.
            packed, gates, _ = adapter.unpack(communication_output, context)
            weighted = (packed.float() * gates[:, None]).to(packed.dtype) if self.npu else packed * gates[:, None]
            communication_output, event = adapter.combine(adapter.prepare_combine(weighted, context), context)
            adapter.wait(event)
        return dict(communication_output=communication_output, compute_output=compute_output)


def autograd_reference(data, limit=0):
    """Independent route-by-expert graph: no adapter, no manual dgrad/wgrad code."""
    x, w1, w2, sw1, sw2 = [data[k].detach().clone().requires_grad_() for k in ("x", "w1", "w2", "sw1", "sw2")]
    routes, gates = data["routes"], data["gates"]
    y = torch.zeros_like(x)
    for e in range(w1.shape[0]):
        rows, slots = torch.nonzero(routes == e, as_tuple=True)
        pre = x[rows] @ w1[e]
        left, right = pre.chunk(2, dim=-1)
        if limit:
            left, right = left.clamp(-limit, limit), right.clamp(-limit, limit)
        z = (torch.nn.functional.silu(left) * right) @ w2[e]
        y = y.index_add(0, rows, z * gates[rows, slots, None])
    left, right = (x @ sw1).chunk(2, dim=-1)
    if limit:
        left, right = left.clamp(-limit, limit), right.clamp(-limit, limit)
    y = y + (torch.nn.functional.silu(left) * right) @ sw2
    dx, g1, g2, sg1, sg2 = torch.autograd.grad((y * data["dy"]).sum(), (x, w1, w2, sw1, sw2))
    return dict(Y=y, dX=dx, expert_dW1=g1, expert_dW2=g2, shared_dW1=sg1, shared_dW2=sg2)


def cpu_check(c):
    from dataclasses import replace
    from .adapters import LocalReference

    # Small, independent float64 mathematical check, explicitly not an NPU result.
    small = replace(
        c,
        tokens=min(c.tokens, 5),
        hidden=min(c.hidden, 8),
        experts=4,
        ep=4,
        topk=min(c.topk, 2),
        width=min(c.width, 4),
        shared_width=min(c.shared_width, 4),
        wgrad_backend="gmm_add",
    )
    layer = Layer(small, LocalReference(small.experts))
    reference = autograd_reference(layer.data, small.clamp)
    checked = []
    with torch.no_grad():
        for name, (a, cached, d) in modes_for_suite(c.layer_suite).items():
            layer.reset()
            actual, _ = layer.step(a, d, overlap_c=cached)
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=1e-10, atol=1e-12)
            checked.append(name)
    return dict(
        scope="CPU_float64_local_routing_math_check_not_EP4_hardware",
        passed=True,
        shape=small.dictionary(),
        checked_modes=checked,
        checked_outputs=list(reference),
        performance_available=False,
    )
