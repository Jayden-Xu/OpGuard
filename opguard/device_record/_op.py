"""Registration of the ``opguard::probe`` custom operator.

The operator's state tensors are explicit (mutable) arguments so compilers see
the writes and keep every probe, in order, because they all mutate ``cursor``.
The record ring itself is not an argument: kernels read its address and
capacity from ``ring_desc`` so the ring can be allocated or replaced after a
CUDA graph was captured.
"""
from __future__ import annotations

import threading
import weakref
from typing import List

import torch

_SCHEMA = (
    "(Tensor[] tensors, int site, Tensor ring_desc, Tensor(a!) cursor, "
    "Tensor(b!) site_counters, Tensor(c!) scratch, Tensor enable, Tensor replay_epoch) -> ()"
)

_registry: dict[int, "weakref.ref"] = {}
_registry_lock = threading.Lock()


def _device_index(device) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"opguard device recorder needs a CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


def register(recorder) -> None:
    """Make ``recorder`` the recorder of its device (one per device)."""
    index = recorder.device.index
    with _registry_lock:
        ref = _registry.get(index)
        existing = ref() if ref is not None else None
        if existing is not None and existing is not recorder:
            raise RuntimeError(
                f"a DeviceRecorder already exists for cuda:{index}; close() it before creating another")
        _registry[index] = weakref.ref(recorder)


def unregister(recorder) -> None:
    index = recorder.device.index
    with _registry_lock:
        ref = _registry.get(index)
        if ref is not None and ref() in (None, recorder):
            del _registry[index]


def lookup(device):
    """The recorder of ``device``; it is used only for host-side bookkeeping."""
    index = _device_index(device)
    with _registry_lock:
        ref = _registry.get(index)
        recorder = ref() if ref is not None else None
    if recorder is None:
        raise RuntimeError(f"opguard::probe: no DeviceRecorder is open on cuda:{index}")
    return recorder


@torch.library.custom_op(
    "opguard::probe", mutates_args={"cursor", "site_counters", "scratch"}, schema=_SCHEMA)
def probe(tensors: List[torch.Tensor], site: int, ring_desc: torch.Tensor, cursor: torch.Tensor,
          site_counters: torch.Tensor, scratch: torch.Tensor, enable: torch.Tensor,
          replay_epoch: torch.Tensor) -> None:
    """Record one call of instrumentation site ``site``.

    Launches one fingerprint kernel per non-empty tensor and one commit kernel,
    all on the current stream. The call id is allocated here, not passed in:
    compilers would bake an int argument into the graph as a constant, while
    this body runs once per eager call and exactly once while a CUDA graph is
    captured (so replayed records carry the id of the captured call).
    """
    lookup(cursor.device)._probe_impl(
        tensors, site, ring_desc, cursor, site_counters, scratch, enable, replay_epoch)


@probe.register_fake
def _probe_fake(tensors, site, ring_desc, cursor, site_counters, scratch, enable, replay_epoch):
    return None
