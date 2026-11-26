"""Four CPU/Gloo ranks check EP partitioning only; never DeepEP or NPU performance.

Run with python -m torch.distributed.run --nproc-per-node=4
--module benchmarks.moe_overlap.cpu_reference --out <new output directory>.
"""

import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.distributed as dist
from benchmarks.moe_overlap.adapters import Context
from benchmarks.moe_overlap.config import Config, MODES
from benchmarks.moe_overlap.layer import Layer, autograd_reference
from benchmarks.moe_overlap.model import initialize


class GlooReference:
    def __init__(self, c, rank):
        self.c, self.rank = c, rank

    def gather(self, x):
        parts = [torch.empty_like(x) for _ in range(4)]
        dist.all_gather(parts, x)
        return torch.cat(parts)

    def dispatch(self, x, routes, gates):
        xx, rr, gg = self.gather(x), self.gather(routes), self.gather(gates)
        local = self.c.experts // 4
        lo = self.rank * local
        rows, slots = torch.nonzero((rr >= lo) & (rr < lo + local), as_tuple=True)
        ids = rr[rows, slots] - lo
        order = torch.argsort(ids, stable=True)
        rows, slots, ids = rows[order], slots[order], ids[order]
        ctx = Context(None, torch.bincount(ids, minlength=local), rows, slots, gg[rows, slots], xx.shape[0])
        return xx[rows], ctx, None

    def wait(self, event):
        pass

    def unpack(self, rx, ctx):
        return rx, ctx.gates, ctx.counts

    def cached(self, dy, ctx):
        return self.gather(dy)[ctx.rows], None

    def pack_gradient(self, dy, ctx):
        return dy

    def prepare_combine(self, x, ctx):
        return x

    def combine(self, x, ctx):
        total = torch.zeros((ctx.received_rows, x.shape[1]), dtype=x.dtype)
        total.index_add_(0, ctx.rows, x)
        dist.all_reduce(total)
        return total[self.rank * self.c.tokens : (self.rank + 1) * self.c.tokens].clone(), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("Four CPU ranks required")
    try:
        c = Config(tokens=3, hidden=4, experts=8, topk=2, width=3, shared_width=3)
        data = []
        for r in range(4):
            d = initialize(c, r, "cpu", torch.float64, all_experts=True)
            # Deliberate uneven routing and one globally empty expert.
            d["routes"].remainder_(7)
            if r == 0:
                d["routes"].zero_()
            data.append(d)
        source = autograd_reference(data[rank])
        global_data = dict(data[0])
        for key in ("x", "dy", "routes", "gates"):
            global_data[key] = torch.cat([d[key] for d in data])
        global_data["sw1"] = torch.zeros_like(global_data["sw1"])
        global_data["sw2"] = torch.zeros_like(global_data["sw2"])
        global_gold = autograd_reference(global_data)
        local = initialize(c, rank, "cpu", torch.float64, all_experts=False)
        local["routes"] = data[rank]["routes"]
        layer = Layer(c, GlooReference(c, rank))
        layer.data = local
        layer.g1 = torch.zeros_like(local["w1"])
        layer.g2 = torch.zeros_like(local["w2"])
        g = c.experts // 4
        source["expert_dW1"] = global_gold["expert_dW1"][rank * g : (rank + 1) * g]
        source["expert_dW2"] = global_gold["expert_dW2"][rank * g : (rank + 1) * g]
        checked = []
        with torch.no_grad():
            for mode, (overlap_a, overlap_d) in MODES.items():
                layer.reset()
                actual, _ = layer.step(overlap_a, overlap_d)
                for key in source:
                    torch.testing.assert_close(actual[key], source[key], rtol=1e-10, atol=1e-12)
                checked.append(mode)
        dest = a.out / f"rank{rank}"
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "result.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    passed=True,
                    checked_modes=checked,
                    scope="CPU_Gloo_EP4_routing_and_gradients_not_DeepEP_NPU",
                    performance_available=False,
                ),
                indent=2,
            )
        )
        dist.barrier()
        if rank == 0:
            print("CPU_GLOO_EP4_MATH_PASSED: 4 ranks, 4 modes, all six outputs; no NPU performance claim", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
