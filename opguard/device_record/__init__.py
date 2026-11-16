"""Device-side call recording for OpGuard (standalone; not wired into tracing yet).

``opguard::probe`` fingerprints CUDA tensors on the GPU and commits one record
per call into a ring in device memory, so CUDA graph replays and compiled code
still produce records. ``DeviceRecorder`` owns the state, reserves ring space,
binds probe counts to captured graphs and reads records back with the snap
protocol (see ``recorder`` for the readback safe-point rule).

No other OpGuard module imports this package, so nothing is loaded or
allocated unless it is used explicitly.
"""
from . import _op
from .recorder import DeviceRecorder, ReadbackHandle, ReadbackResult

probe = _op.probe

__all__ = ["DeviceRecorder", "ReadbackHandle", "ReadbackResult", "probe"]
