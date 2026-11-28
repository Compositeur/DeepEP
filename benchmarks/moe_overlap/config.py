"""Standard-library configuration and plan generation."""

from dataclasses import asdict, dataclass
from .scenarios import WINDOWS, modes_for_suite, layer_metrics

MODES = {"serial": (False, False), "d_only": (False, True), "a_only": (True, False), "ad": (True, True)}
REFERENCE_DEEPEP_REVISION = "83252a54d2363734a706a7dd8368bebc3094af5a"


@dataclass(frozen=True)
class Config:
    tokens: int = 4096
    hidden: int = 2816
    experts: int = 256
    topk: int = 8
    width: int = 768
    shared_width: int = 768
    ep: int = 4
    api: str = "v1_global_clos"
    num_sms: int = 32
    wgrad_backend: str = "gmm_add"
    seed: int = 20260823
    warmup: int = 5
    iterations: int = 30
    repeats: int = 3
    profile_iterations: int = 0
    clamp: float = 0.0
    precision_rtol: float = 0.02
    precision_atol: float = 0.002
    layer_suite: str = "AD"
    attention_tokens: int = 32
    attention_heads: int = 4
    attention_head_dim: int = 16
    recompute_experts: bool = False
    source_reference_dtype: str = "fp32"
    weight_init_std: float = 0.1
    dispatch_num_sms: int = 0
    combine_num_sms: int = 0
    reduce_num_sms: int = 0

    def validate(self):
        import math

        for k in (
            "tokens",
            "hidden",
            "experts",
            "topk",
            "width",
            "shared_width",
            "ep",
            "iterations",
            "repeats",
            "attention_tokens",
            "attention_heads",
            "attention_head_dim",
        ):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(k + " must be a positive integer")
        if self.ep not in (4, 16) or self.experts % self.ep or self.topk > self.experts:
            raise ValueError("Require EP4/EP16, divisible expert count and valid topk")
        if type(self.recompute_experts) is not bool:
            raise ValueError("recompute_experts must be boolean")
        if self.source_reference_dtype not in ("fp32", "bf16"):
            raise ValueError("source_reference_dtype must be fp32 or bf16")
        if (
            type(self.weight_init_std) not in (int, float)
            or not math.isfinite(self.weight_init_std)
            or self.weight_init_std <= 0
        ):
            raise ValueError("weight_init_std must be finite and positive")
        if self.iterations < 2 or self.repeats < 2:
            raise ValueError("At least two iterations and two independent repeats required")
        for k in ("warmup", "profile_iterations"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(k)
        if self.api not in ("v1_global_clos", "v2_elastic"):
            raise ValueError("Unsupported communication API")
        modes_for_suite(self.layer_suite)
        if self.wgrad_backend not in ("gmm_add", "pointer"):
            raise ValueError("Unsupported wgrad")
        if type(self.num_sms) is not int or self.num_sms < 0 or self.num_sms % 2:
            raise ValueError("num_sms must be nonnegative and even")
        if self.api == "v2_elastic" and self.num_sms != 0:
            raise ValueError("V2 uses version defaults: set num_sms=0; no shared V1 SMS assumption")
        for key in ("dispatch_num_sms", "combine_num_sms", "reduce_num_sms"):
            value = getattr(self, key)
            if type(value) is not int or value < 0 or value % 2:
                raise ValueError(key + " must be nonnegative and even")
            if self.api != "v2_elastic" and value:
                raise ValueError("Independent communication core controls require V2")
        for k in ("clamp", "precision_rtol", "precision_atol"):
            v = getattr(self, k)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(k)
        return self

    def dictionary(self):
        return asdict(self)


def plan(c):
    c.validate()
    return dict(
        scope="real single MoE layer, fixed routes/gates; not full training",
        configuration=c.dictionary(),
        reference_deepep_revision=REFERENCE_DEEPEP_REVISION,
        cases=[*WINDOWS, "layer", "all"],
        windows=WINDOWS,
        modes={
            k: dict(overlap_a=a, overlap_c=cached, overlap_d=d)
            for k, (a, cached, d) in modes_for_suite(c.layer_suite).items()
        },
        metrics=list(layer_metrics(c.layer_suite)),
        correctness=["Y", "dX", "expert_dW1", "expert_dW2", "shared_dW1", "shared_dW2"],
        gates="frozen; no router parameter gradient coverage",
        interfaces="V1/V2 separate runs and separate baselines",
    )
