"""A/D switches change waits, never expert operands or arithmetic."""

import torch
from .model import Math, initialize, expert_forward, expert_dgrad, expert_wgrad, shared_forward, shared_backward


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

    def reset(self):
        self.g1.zero_()
        self.g2.zero_()

    def step(self, overlap_a=False, overlap_d=False, timed=False):
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
        with torch.profiler.record_function("moe/backward_cached_dispatch"):
            dy, done = adapter.cached(p["dy"], ctx)
            adapter.wait(done)
            dy = adapter.pack_gradient(dy, ctx)
        with torch.profiler.record_function("moe/experts_dgrad"):
            dxpacked, wctx = expert_dgrad(self.math, dy, gates, p["w1t"], p["w2t"], counts, ectx)
            dxshared, sg1, sg2 = shared_backward(p["dy"], p["sw1"], p["sw2"], sctx)
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
                [("total_us", "start", "end"), ("a_window_us", "a_start", "a_end"), ("d_window_us", "d_start", "d_end")]
            )
            if clock
            else {}
        )
        return outputs, times

    def window(self, case, control):
        """D uses a real forward/backward-produced fixture prepared outside timing."""
        if case not in ("A", "D") or control not in ("comm_only", "compute_only", "serial", "overlap"):
            raise ValueError("Unknown window/control")
        p = self.data
        adapter = self.adapter
        clock = Clock()
        clock.mark("start")
        comm = control != "compute_only"
        compute = control != "comm_only"
        if case == "A":
            done = None
            if comm:
                received, context, done = adapter.dispatch(p["x"], p["routes"], p["gates"])
            if control == "serial":
                adapter.wait(done)
            if compute:
                shared_forward(p["x"], p["sw1"], p["sw2"], self.c.clamp)
        else:
            if self.last_d is None:
                raise RuntimeError("Prepare D fixture with a complete step first")
            payload, ctx, wctx, counts = self.last_d
            done = None
            if comm:
                _, done = adapter.combine(payload, ctx)
            if control == "serial":
                adapter.wait(done)
            if compute:
                expert_wgrad(self.math, wctx, counts, self.g1, self.g2)
        if comm and control != "serial":
            adapter.wait(done)
        clock.mark("end")
        return clock.results([("window_us", "start", "end")])["window_us"]


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
    from .config import MODES

    # Small, independent float64 mathematical check, explicitly not an NPU result.
    small = replace(
        c,
        tokens=min(c.tokens, 5),
        hidden=min(c.hidden, 8),
        experts=4,
        topk=min(c.topk, 2),
        width=min(c.width, 4),
        shared_width=min(c.shared_width, 4),
        wgrad_backend="gmm_add",
    )
    layer = Layer(small, LocalReference(small.experts))
    reference = autograd_reference(layer.data, small.clamp)
    checked = []
    with torch.no_grad():
        for name, (a, d) in MODES.items():
            layer.reset()
            actual, _ = layer.step(a, d)
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
