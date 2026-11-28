"""V2 communication controls keep reduction independent of dispatch channels."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.moe_overlap.config import Config


@pytest.mark.parametrize("cores", [8, 16, 32])
def test_v2_dispatch_combine_cores_and_fixed_reduce(cores):
    torch = pytest.importorskip("torch")
    from benchmarks.moe_overlap.adapters import DeepEP

    calls = []

    class Event:
        def current_stream_wait(self):
            pass

    class Elastic:
        def __init__(self, **kwargs):
            pass

        def capture(self):
            return Event()

        def dispatch(self, x, handle=None, **kwargs):
            calls.append(("dispatch", handle, kwargs))
            if handle is None:
                handle = SimpleNamespace(num_sms=kwargs["num_sms"])
            else:
                assert kwargs["do_expand"] is True
                assert "num_sms" not in kwargs  # cached operation inherits handle channels
            return x, None, None, handle, Event()

        def combine(self, x, handle, **kwargs):
            calls.append(("combine", handle, kwargs))
            assert kwargs["combine_num_sms"] == cores
            assert kwargs["reduce_num_sms"] == 32
            assert "num_sms" not in kwargs
            return x, None, Event()

        def destroy(self):
            pass

    c = Config(api="v2_elastic", num_sms=0, dispatch_num_sms=cores, combine_num_sms=cores, reduce_num_sms=32)
    with patch.dict(sys.modules, {"deep_ep": SimpleNamespace(Buffer=object, ElasticBuffer=Elastic)}):
        adapter = DeepEP(c, object())
        x = torch.zeros((2, 4))
        _, ctx, _ = adapter.dispatch(x, torch.zeros((2, 2), dtype=torch.int64), torch.ones((2, 2)))
        adapter.cached(x, ctx)
        adapter.combine(x, ctx)
        assert adapter.core_counts == dict(dispatch=cores, combine=cores, reduce=32)
        assert calls[0][2]["num_sms"] == cores
        assert all(call[2]["allocate_on_comm_stream"] for call in calls)
        adapter.destroy()


@pytest.mark.parametrize("field", ["dispatch_num_sms", "combine_num_sms", "reduce_num_sms"])
@pytest.mark.parametrize("value", [-2, 3, True])
def test_core_controls_reject_invalid_counts(field, value):
    with pytest.raises(ValueError):
        Config(api="v2_elastic", num_sms=0, **{field: value}).validate()


def test_v1_does_not_claim_independent_reduce_control():
    with pytest.raises(ValueError, match="require V2"):
        Config(reduce_num_sms=32).validate()
