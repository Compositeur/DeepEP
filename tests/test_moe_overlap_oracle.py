"""BF16 arithmetic boundaries and dtype-specific correctness reporting."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.moe_overlap.config import Config


def test_bf16_activation_backward_matches_autograd():
    torch = pytest.importorskip("torch")
    from benchmarks.moe_overlap.model import activation, activation_backward

    generator = torch.Generator().manual_seed(37)
    for clamp in (0, 1.0):
        x = (torch.randn((16, 128), generator=generator) * 2).bfloat16().requires_grad_()
        dy = torch.randn((16, 64), generator=generator).bfloat16()
        y, ctx = activation(x, clamp)
        expected = torch.autograd.grad((y * dy).sum(), x)[0]
        actual = activation_backward(dy, ctx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("weight_std", [0.1, 0.006])
def test_sampled_oracle_reconstructs_weights_and_honors_dtype(weight_std):
    torch = pytest.importorskip("torch")
    from benchmarks.moe_overlap.model import initialize
    from benchmarks.moe_overlap.oracle import sampled_reference, reference_errors

    c = Config(
        tokens=3,
        hidden=64,
        experts=8,
        topk=2,
        width=32,
        shared_width=32,
        source_reference_dtype="bf16",
        weight_init_std=weight_std,
    )
    data = initialize(c, 0, "cpu", torch.bfloat16, all_experts=True)
    x, w1, w2, sw1, sw2 = [data[key].clone().requires_grad_() for key in ("x", "w1", "w2", "sw1", "sw2")]
    contributions = []
    # Independent full graph; cast routing reduction at the BF16 boundary.
    for token in range(c.tokens):
        terms = []
        for slot in range(c.topk):
            expert = int(data["routes"][token, slot])
            gate = data["gates"][token, slot]
            left, right = (x[token] @ w1[expert]).chunk(2)
            z = (torch.nn.functional.silu(left) * right) @ w2[expert]
            terms.append((z.float() * gate).bfloat16().float())
        contributions.append(torch.stack(terms).sum(0).bfloat16())
    left, right = (x @ sw1).chunk(2, dim=-1)
    y = torch.stack(contributions) + (torch.nn.functional.silu(left) * right) @ sw2
    dx = torch.autograd.grad((y * data["dy"]).sum(), x)[0]
    reference = sampled_reference(SimpleNamespace(c=c, data=data), 0)
    actual = dict(Y=y.detach(), dX=dx.detach())
    assert reference_errors(actual, reference)["passed"]
    for token, gold_y, _ in reference:
        torch.testing.assert_close(y[token].float(), gold_y, rtol=0, atol=0)
    changed = dict(actual, Y=actual["Y"] + 1)
    assert not reference_errors(changed, reference)["passed"]
    fp32 = sampled_reference(SimpleNamespace(c=c, data=data), 0, "fp32")
    assert any(not torch.equal(a[1], b[1]) for a, b in zip(reference, fp32))


def test_source_reference_dtype_is_explicit_and_default_stays_fp32():
    assert Config().source_reference_dtype == "fp32"
    with pytest.raises(ValueError):
        replace(Config(), source_reference_dtype="automatic").validate()


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), True])
def test_invalid_weight_initialization_is_rejected(value):
    with pytest.raises(ValueError):
        replace(Config(), weight_init_std=value).validate()
