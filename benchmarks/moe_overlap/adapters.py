"""V1 public Global Clos / V2 public Elastic, with one canonical expert-row view."""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    handle: object
    counts: object = None
    rows: object = None
    slots: object = None
    gates: object = None
    received_rows: int = 0


class LocalReference:
    """CPU-only routing transport, for math checks; never an NPU performance result."""

    def __init__(self, experts):
        self.experts = experts

    def dispatch(self, x, routes, gates):
        rows, slots = torch.nonzero(routes >= 0, as_tuple=True)
        ids = routes[rows, slots]
        order = torch.argsort(ids, stable=True)
        rows, slots, ids = rows[order], slots[order], ids[order]
        ctx = Context(None, torch.bincount(ids, minlength=self.experts), rows, slots, gates[rows, slots], x.shape[0])
        return x[rows], ctx, None

    def wait(self, event):
        pass

    def unpack(self, rx, ctx):
        return rx, ctx.gates, ctx.counts

    def cached(self, dy, ctx):
        return dy[ctx.rows], None

    def pack_gradient(self, dy, ctx):
        return dy

    def prepare_combine(self, x, ctx):
        return x

    def combine(self, x, ctx):
        out = torch.zeros((ctx.received_rows, x.shape[-1]), device=x.device, dtype=x.dtype)
        out.index_add_(0, ctx.rows, x)
        return out, None

    def destroy(self):
        pass


class DeepEP:
    def __init__(self, c, group):
        from deep_ep import Buffer, ElasticBuffer

        self.c = c
        self.v1 = c.api == "v1_global_clos"
        self.buffer = None
        if self.v1:
            # Same public factory and workspace queries as PR 217; no transport override.
            dc, cc = Buffer.get_dispatch_config(c.ep), Buffer.get_combine_config(c.ep)
            hidden_bytes = c.hidden * 2
            nvl = max(x.get_nvl_buffer_size_hint(hidden_bytes, c.ep) for x in (dc, cc))
            rdma = max(x.get_rdma_buffer_size_hint(hidden_bytes, c.ep) for x in (dc, cc))
            self.buffer = Buffer(group, nvl, rdma, machine_type="global_clos", explicitly_destroy=True)
            cls = type(self.buffer)
            cls.set_num_sms(c.num_sms)
            self.dc, self.cc = cls.get_dispatch_config(c.ep), cls.get_combine_config(c.ep)
        else:
            # Force HT even for small smoke inputs: LL has different training semantics.
            self.buffer = ElasticBuffer(
                group=group,
                num_max_tokens_per_rank=max(c.tokens, 257),
                hidden=c.hidden,
                num_topk=c.topk,
                explicitly_destroy=True,
                machine_type="global_clos",
            )

    def wait(self, event):
        if event is None:
            raise RuntimeError("Expected an asynchronous completion event")
        event.current_stream_wait()

    def dispatch(self, x, routes, gates):
        ready = self.buffer.capture()
        if self.v1:
            nr, nrdma, ne, mask, event = self.buffer.get_dispatch_layout(
                routes, self.c.experts, previous_event=ready, async_finish=True
            )
            rx, ri, rw, counts, handle, done = self.buffer.dispatch(
                x,
                topk_idx=routes,
                topk_weights=gates,
                num_tokens_per_rank=nr,
                num_tokens_per_rdma_rank=nrdma,
                num_tokens_per_expert=ne,
                is_token_in_rank=mask,
                config=self.dc,
                previous_event=event,
                async_finish=True,
            )
            return rx, Context(handle, counts, ri, rw), done
        rx, _, rw, handle, done = self.buffer.dispatch(
            x,
            topk_idx=routes,
            topk_weights=gates,
            num_experts=self.c.experts,
            do_expand=True,
            expert_alignment=1,
            previous_event=ready,
            async_with_compute_stream=True,
        )
        return rx, Context(handle, gates=rw), done

    def unpack(self, rx, ctx):
        if self.v1:
            ri, rw = ctx.rows, ctx.slots
            rows, slots = torch.nonzero(ri >= 0, as_tuple=True)
            ids = ri[rows, slots].long()
            order = torch.argsort(ids, stable=True)
            rows, slots, ids = rows[order], slots[order], ids[order]
            ctx.rows, ctx.slots = rows, slots
            ctx.gates = rw[rows, slots]
            ctx.received_rows = rx.shape[0]
            ctx.counts = torch.bincount(ids, minlength=self.c.experts // self.c.ep)
            return rx[rows], ctx.gates, ctx.counts
        counts = ctx.handle.num_recv_tokens_per_expert_list
        ctx.counts = torch.tensor(counts, dtype=torch.int64, device=rx.device)
        if sum(counts) != rx.shape[0]:
            raise RuntimeError("Unexpected expanded expert layout")
        return rx, ctx.gates, ctx.counts

    def cached(self, dy, ctx):
        ready = self.buffer.capture()
        if self.v1:
            rx, _, _, _, _, done = self.buffer.dispatch(
                dy, handle=ctx.handle, config=self.dc, previous_event=ready, async_finish=True
            )
        else:
            rx, _, _, _, done = self.buffer.dispatch(
                dy, handle=ctx.handle, previous_event=ready, do_cpu_sync=False, async_with_compute_stream=True
            )
        return rx, done

    def pack_gradient(self, dy, ctx):
        return dy[ctx.rows] if self.v1 else dy

    def prepare_combine(self, x, ctx):
        if not self.v1:
            return x.contiguous()
        out = torch.zeros((ctx.received_rows, x.shape[-1]), device=x.device, dtype=torch.float32)
        out.index_add_(0, ctx.rows, x.float())
        return out.to(x.dtype)

    def combine(self, x, ctx):
        # All dtype/layout conversions precede capture: avoid producer/consumer races.
        if not x.is_contiguous():
            raise ValueError("Prepare contiguous combine input before capture")
        ready = self.buffer.capture()
        if self.v1:
            y, _, done = self.buffer.combine(x, ctx.handle, config=self.cc, previous_event=ready, async_finish=True)
        else:
            y, _, done = self.buffer.combine(x, ctx.handle, previous_event=ready, async_with_compute_stream=True)
        return y, done

    def destroy(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
