"""Explicit SwiGLU dgrad/deferred wgrad; independent torch autograd is the oracle."""

import torch


class Math:
    def __init__(self, npu=False, wgrad_backend="gmm_add"):
        self.npu = npu
        self.pointer = None
        if npu:
            import torch_npu

            self.ops = torch_npu
        if wgrad_backend == "pointer":
            if not npu:
                raise ValueError("pointer requires NPU")
            try:
                from pointer_gmm import npu_grouped_matmul_add_ptr_
            except ImportError as exc:
                raise RuntimeError("pointer was explicitly selected but pointer_gmm is unavailable") from exc
            self.pointer = npu_grouped_matmul_add_ptr_

    def grouped(self, x, w, counts):
        if self.npu:
            return self.ops.npu_grouped_matmul(
                [x], [w], group_list=counts, group_type=0, group_list_type=1, split_item=3
            )[0]
        out = []
        offset = 0
        for e, n in enumerate(counts.tolist()):
            out.append(x[offset : offset + n] @ w[e])
            offset += n
        return torch.cat(out, dim=0)

    def wgrad(self, x, output_grad, counts, target):
        if self.pointer is not None:
            self.pointer(x.contiguous(), output_grad.contiguous(), counts, tuple(target.unbind()))
        elif self.npu:
            dw = self.ops.npu_grouped_matmul(
                [x.t()], [output_grad], group_list=counts, group_type=2, group_list_type=1, split_item=3
            )[0]
            target.add_(dw.reshape_as(target))
        else:
            offset = 0
            for e, n in enumerate(counts.tolist()):
                target[e].add_(x[offset : offset + n].t() @ output_grad[offset : offset + n])
                offset += n


def activation(pre, limit=0):
    left, right = pre.chunk(2, dim=-1)
    a = left.clamp(-limit, limit) if limit else left
    b = right.clamp(-limit, limit) if limit else right
    return torch.nn.functional.silu(a) * b, (left, right, a, b, limit)


def activation_backward(dact, ctx):
    left, right, a, b, limit = ctx
    # Match the backward of the actual SiLU operation. Expanding the derivative
    # into BF16 sigmoid/multiply/add operations introduces extra rounding steps.
    dl = torch.ops.aten.silu_backward(dact * b, a)
    dr = dact * torch.nn.functional.silu(a)
    if limit:
        dl = dl * ((left >= -limit) & (left <= limit))
        dr = dr * ((right >= -limit) & (right <= limit))
    return torch.cat((dl, dr), dim=-1)


def expert_forward(math, x, w1, w2, counts, limit=0):
    pre = math.grouped(x, w1, counts)
    act, ctx = activation(pre, limit)
    z = math.grouped(act, w2, counts)
    return z, (x, act, ctx)


def expert_dgrad(math, dy, gates, w1t, w2t, counts, ctx):
    x, act, actctx = ctx
    dz = (dy.float() * gates[:, None]).to(dy.dtype) if dy.dtype == torch.bfloat16 else dy * gates[:, None]
    dact = math.grouped(dz, w2t, counts)
    dpre = activation_backward(dact, actctx)
    dx = math.grouped(dpre, w1t, counts)
    return dx, (x, dpre, act, dz)


def expert_wgrad(math, ctx, counts, g1, g2):
    x, dpre, act, dz = ctx
    math.wgrad(act, dz, counts, g2)
    math.wgrad(x, dpre, counts, g1)


def shared_forward(x, w1, w2, limit=0):
    act, ctx = activation(x @ w1, limit)
    return act @ w2, (x, act, ctx)


def shared_backward(dy, w1, w2, ctx):
    x, act, actctx = ctx
    dpre = activation_backward(dy @ w2.t(), actctx)
    dx, g1, g2 = dpre @ w1.t(), x.t() @ dpre, act.t() @ dy
    if dy.dtype == torch.bfloat16:
        g1, g2 = g1.float(), g2.float()
    return dx, g1, g2


def attention_replay(x, qkv_weight, out_weight, heads):
    """Eager causal attention on saved input, used only by isolated R replay."""
    tokens = x.shape[0]
    q, k, v = (x @ qkv_weight).chunk(3, dim=-1)
    head_dim = q.shape[-1] // heads
    q, k, v = [part.reshape(tokens, heads, head_dim).transpose(0, 1) for part in (q, k, v)]
    scores = (q @ k.transpose(-1, -2)) * head_dim**-0.5
    causal = torch.ones((tokens, tokens), dtype=torch.bool, device=x.device).triu(1)
    probabilities = scores.float().masked_fill(causal, float("-inf")).softmax(dim=-1).to(x.dtype)
    context = (probabilities @ v).transpose(0, 1).reshape(tokens, heads * head_dim)
    return context @ out_weight


def initialize(c, rank, device, dtype, all_experts=False):
    """Global expert seeds let the source oracle reconstruct remote weights."""

    def rand(shape, seed, scale=0.1):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(shape, generator=g) * scale).to(device=device, dtype=dtype)

    n = c.experts if all_experts else c.experts // c.ep
    first = 0 if all_experts else rank * n
    w1 = torch.stack(
        [rand((c.hidden, 2 * c.width), c.seed + 10000 + e, c.weight_init_std) for e in range(first, first + n)]
    )
    w2 = torch.stack(
        [rand((c.width, c.hidden), c.seed + 20000 + e, c.weight_init_std) for e in range(first, first + n)]
    )
    sw1 = rand((c.hidden, 2 * c.shared_width), c.seed + 30000 + rank, c.weight_init_std)
    sw2 = rand((c.shared_width, c.hidden), c.seed + 40000 + rank, c.weight_init_std)
    x = rand((c.tokens, c.hidden), c.seed + rank)
    dy = rand((c.tokens, c.hidden), c.seed + 1000 + rank)
    g = torch.Generator().manual_seed(c.seed + 2000 + rank)
    routes = torch.stack([torch.randperm(c.experts, generator=g)[: c.topk] for _ in range(c.tokens)]).to(device)
    gates = torch.full(
        (c.tokens, c.topk), 1 / c.topk, device=device, dtype=torch.float32 if dtype == torch.bfloat16 else dtype
    )
    return dict(
        x=x,
        dy=dy,
        routes=routes,
        gates=gates,
        w1=w1,
        w2=w2,
        sw1=sw1,
        sw2=sw2,
        w1t=w1.transpose(1, 2).contiguous(),
        w2t=w2.transpose(1, 2).contiguous(),
    )
