"""Standard-library configuration and plan generation."""

from dataclasses import asdict, dataclass

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

    def validate(self):
        import math

        for k in ("tokens", "hidden", "experts", "topk", "width", "shared_width", "ep", "iterations", "repeats"):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(k + " must be a positive integer")
        if self.ep != 4 or self.experts % self.ep or self.topk > self.experts:
            raise ValueError("Require EP4, divisible expert count and valid topk")
        if self.iterations < 2 or self.repeats < 2:
            raise ValueError("At least two iterations and two independent repeats required")
        for k in ("warmup", "profile_iterations"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(k)
        if self.api not in ("v1_global_clos", "v2_elastic"):
            raise ValueError("Unsupported communication API")
        if self.wgrad_backend not in ("gmm_add", "pointer"):
            raise ValueError("Unsupported wgrad")
        if type(self.num_sms) is not int or self.num_sms < 0 or self.num_sms % 2:
            raise ValueError("num_sms must be nonnegative and even")
        if self.api == "v2_elastic" and self.num_sms != 0:
            raise ValueError("V2 uses version defaults: set num_sms=0; no shared V1 SMS assumption")
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
        cases=["D", "A", "layer", "all"],
        modes={k: dict(overlap_a=a, overlap_d=d) for k, (a, d) in MODES.items()},
        metrics=["a_window_us", "d_window_us", "total_us"],
        correctness=["Y", "dX", "expert_dW1", "expert_dW2", "shared_dW1", "shared_dW2"],
        gates="frozen; no router parameter gradient coverage",
        interfaces="V1/V2 separate runs and separate baselines",
    )
