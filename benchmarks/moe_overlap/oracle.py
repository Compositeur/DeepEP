"""Independent source-token autograd references with explicit arithmetic dtype."""

import torch


def sampled_reference(layer, rank, reference_dtype=None):
    """Reconstruct global expert weights without transport handles or row maps."""
    c, data = layer.c, layer.data
    selected = reference_dtype or c.source_reference_dtype
    dtype = torch.bfloat16 if selected == "bf16" else torch.float32

    def weights(expert):
        first = torch.Generator().manual_seed(c.seed + 10000 + expert)
        second = torch.Generator().manual_seed(c.seed + 20000 + expert)
        return (
            (torch.randn((c.hidden, 2 * c.width), generator=first) * c.weight_init_std).bfloat16().to(dtype),
            (torch.randn((c.width, c.hidden), generator=second) * c.weight_init_std).bfloat16().to(dtype),
        )

    def forward(x, w1, w2):
        left, right = (x @ w1).chunk(2, dim=-1)
        if c.clamp:
            left, right = left.clamp(-c.clamp, c.clamp), right.clamp(-c.clamp, c.clamp)
        return (torch.nn.functional.silu(left) * right) @ w2

    result = []
    for token in sorted({0, c.tokens // 2, c.tokens - 1}):
        x, dy = data["x"][token].cpu().to(dtype), data["dy"][token].cpu().to(dtype)
        routed_y, routed_dx = torch.zeros(c.hidden), torch.zeros(c.hidden)
        for expert, gate in zip(data["routes"][token].cpu().tolist(), data["gates"][token].cpu().tolist()):
            w1, w2 = weights(expert)
            xx = x.clone().requires_grad_()
            y = forward(xx, w1, w2)
            y = (y.float() * gate).to(dtype)
            dx = torch.autograd.grad((y * dy).sum(), xx)[0]
            # The routing reduction accumulates BF16 contributions in FP32,
            # then returns BF16 before the shared branch is added.
            routed_y += y.detach().float()
            routed_dx += dx.detach().float()
        xx = x.clone().requires_grad_()
        shared = forward(xx, data["sw1"].cpu().to(dtype), data["sw2"].cpu().to(dtype))
        shared_dx = torch.autograd.grad((shared * dy).sum(), xx)[0]
        y = (routed_y.to(dtype) + shared.detach()).to(dtype)
        dx = (routed_dx.to(dtype) + shared_dx.detach()).to(dtype)
        result.append((token, y.float(), dx.float()))
    return result


def reference_errors(actual, reference, *, rtol=0.06, atol=0.002):
    """Record deviations, including failures, independently of the selected gate."""
    rows = []
    for token, y, dx in reference:
        row = dict(token=token)
        for name, gold in (("Y", y), ("dX", dx)):
            value = actual[name][token].detach().cpu().float()
            error = (value - gold).abs()
            mismatched = error > atol + rtol * gold.abs()
            finite = bool(torch.isfinite(value).all() and torch.isfinite(gold).all())
            row[name] = dict(
                max_abs=float(error.max()),
                mismatched=int(mismatched.sum()),
                elements=value.numel(),
                finite=finite,
                normalized_rmse=float(error.square().mean().sqrt() / gold.square().mean().sqrt().clamp_min(1e-30)),
            )
        rows.append(row)
    return dict(
        rtol=rtol,
        atol=atol,
        samples=rows,
        passed=all(item[key]["finite"] and item[key]["mismatched"] == 0 for item in rows for key in ("Y", "dX")),
    )
