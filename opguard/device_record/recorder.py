"""Device-side call recorder: state in GPU memory, snap readback on the host.

Each ``opguard::probe`` call fingerprints its CUDA tensors on the GPU and
commits one slot of a record ring in device memory. Because recording is GPU
work inside an operator, CUDA graph replays re-run it. The host reserves ring
space before issuing work (so records are normally never overwritten) and
reads rings back with the snap protocol; every call whose slot is not
accepted becomes a placeholder.

Two rings alternate. A readback first switches ``ring_desc`` to the other ring
on the compute stream and then records the event the copy waits for, so the
ring being read is never written again: commits after the event go to the
other ring, and slots overwritten by a later lap before the event fail the
``seq == s`` check. Growing the ring is the same switch.

Readback safe point
-------------------
A readback (explicit, or automatic when space is needed) snapshots the
counters on the caller's current stream ``S``. It is only correct when every
stream that issued probes has been joined into ``S`` (pass such streams as
``join_streams`` or join them yourself) and when probes later issued on other
streams are ordered after this point of ``S`` (the usual fork/join pattern
where side streams fork from an event on ``S`` and join back into it). The
snapshot only orders work within one stream; outside a safe point the counter
value is not a "fully written" boundary. CUDA graph replay satisfies this,
since capture requires every forked stream to join before capture ends.
Automatic readbacks triggered by eager probes assume the caller is at a safe
point; establishing that for multi-stream eager code is the caller's job.
"""
from __future__ import annotations

import collections
import contextlib
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from . import _op

SLOT_HEADER_WORDS = 6  # seq, site, n, k, call_id, replay_epoch
MIN_EAGER_CAPACITY = 64


def _load_extension():
    try:
        import xor_reduce_cuda as ext
    except Exception as exc:  # pragma: no cover - depends on the build
        raise RuntimeError(
            "opguard.device_record needs the native xor_reduce_cuda extension; "
            "rebuild OpGuard with OPGUARD_TRACING_BUILD=1") from exc
    if not hasattr(ext, "device_record_probe"):  # pragma: no cover - stale build
        raise RuntimeError("the installed xor_reduce_cuda extension predates opguard.device_record; rebuild OpGuard")
    return ext


# --------------------------------------------------------------------- results

@dataclass
class ReadbackResult:
    """Records accepted by one or more readbacks, plus placeholders for lost calls.

    ``fingerprints[i, j]`` is the 4-word fingerprint of tensor ``j`` of record
    ``i``; entries ``j >= n[i]`` are zero. Placeholders identify lost calls by
    ``(site, k)``; ``lost`` (calls not accepted) always equals
    ``len(placeholder_k)``. ``written_during_copy`` is statistics only: the
    number of commits that happened between the snapshot and the end of the
    copy (summed ``c1 - c0``); those went to the other ring.
    """

    s: np.ndarray
    site: np.ndarray
    n: np.ndarray
    k: np.ndarray
    call_id: np.ndarray
    replay_epoch: np.ndarray
    fingerprints: np.ndarray
    placeholder_site: np.ndarray
    placeholder_k: np.ndarray
    lost: int = 0
    written_during_copy: int = 0
    readbacks: int = 0

    _RECORD_FIELDS = ("s", "site", "n", "k", "call_id", "replay_epoch", "fingerprints")

    def __len__(self) -> int:
        return int(self.s.shape[0])

    def __getitem__(self, index) -> "ReadbackResult":
        """Select records (mask, slice or indices); placeholders and counters are kept."""
        picked = {name: getattr(self, name)[index] for name in self._RECORD_FIELDS}
        return ReadbackResult(**picked, placeholder_site=self.placeholder_site,
                              placeholder_k=self.placeholder_k, lost=self.lost,
                              written_during_copy=self.written_during_copy, readbacks=self.readbacks)

    @classmethod
    def empty(cls, t_max: int) -> "ReadbackResult":
        z = np.zeros(0, dtype=np.int64)
        return cls(z, z, z, z, z, z, np.zeros((0, t_max, 4), dtype=np.uint32), z, z)

    @classmethod
    def concat(cls, results: Sequence["ReadbackResult"], t_max: int) -> "ReadbackResult":
        if not results:
            return cls.empty(t_max)
        if len(results) == 1:
            return results[0]
        cat = lambda name: np.concatenate([getattr(r, name) for r in results])
        return cls(*(cat(name) for name in cls._RECORD_FIELDS),
                   placeholder_site=cat("placeholder_site"), placeholder_k=cat("placeholder_k"),
                   lost=sum(r.lost for r in results),
                   written_during_copy=sum(r.written_during_copy for r in results),
                   readbacks=sum(r.readbacks for r in results))


# --------------------------------------------------------------- ring blocks

class _Block:
    """One record ring with everything a readback of it needs.

    ``desc`` is this ring's own device-side ``[address, capacity]``; switching
    copies it into ``ring_desc`` device-to-device on the compute stream.
    ``snap`` (device) and the pinned ``host``/``snap_host``/``c1_host`` belong to
    this ring's readback. At most one readback of a ring is pending at a time:
    the ring is only switched back to after that readback was collected.
    """

    __slots__ = ("ring", "desc", "snap", "host", "snap_host", "c1_host", "cap", "pending")

    def __init__(self, cap: int, slot_words: int, num_sites: int, device):
        i64 = dict(dtype=torch.int64, device=device)
        self.ring = torch.zeros((cap, slot_words), **i64)
        self.ring[:, 0] = -1  # otherwise slot 0 would look like a valid record number 0
        self.desc = torch.tensor([self.ring.data_ptr(), cap], **i64)
        self.snap = torch.zeros(1 + num_sites, **i64)
        self.host = torch.empty((cap, slot_words), dtype=torch.int64, pin_memory=True)
        self.snap_host = torch.empty(1 + num_sites, dtype=torch.int64, pin_memory=True)
        self.c1_host = torch.empty(1, dtype=torch.int64, pin_memory=True)
        self.cap = cap
        self.pending: Optional[ReadbackHandle] = None


class ReadbackHandle:
    """An issued readback. Collect it with ``DeviceRecorder.collect``."""

    __slots__ = ("block", "event", "done")

    def __init__(self, block: Optional[_Block], event=None):
        self.block, self.event = block, event
        self.done = block is None  # nothing to read before any ring exists


# ------------------------------------------------------------------- recorder

class DeviceRecorder:
    """Per-device recorder that owns the state tensors used by ``opguard::probe``.

    Args:
        device: CUDA device.
        tensors_per_site: ``T[i]``, the maximum number of tensors site ``i`` probes.
        capacity: initial capacity of each of the two rings, in slots; 0
            allocates on first need (which may be after graphs were captured).
        replays_per_readback: when space is allocated for a graph replay, size
            the ring for this many replays.
        capture_wait_timeout: seconds an operation that needs readback or
            allocation waits for another thread's capture to finish.
        overflow_check: reserve space before issuing work (default). Only tests
            turn this off, to exercise the placeholder fallback.
        on_call: optional ``on_call(site, call_id, tensors, capturing)``,
            invoked (without the recorder lock) before the call's kernels are
            issued. It may itself call ``probe``. If the probe fails after the
            callback returned (for example, waiting for another thread's
            capture timed out), the call produces no record and the exception
            propagates to the caller.

    All state except the rings is allocated here, before any capture, and is
    only ever modified in place afterwards: captured graphs keep its addresses.
    """

    def __init__(self, device, tensors_per_site: Sequence[int], capacity: int = 0,
                 replays_per_readback: int = 1, capture_wait_timeout: float = 60.0,
                 overflow_check: bool = True, on_call: Optional[Callable] = None):
        self.device = torch.device("cuda", _op._device_index(device))
        T = [int(t) for t in tensors_per_site]
        if not T or min(T) < 0:
            raise ValueError("tensors_per_site must be a non-empty list of non-negative ints")
        if int(capacity) < 0 or int(replays_per_readback) < 1:
            raise ValueError("capacity must be >= 0 and replays_per_readback >= 1")
        self._ext = _load_extension()
        self.tensors_per_site = T
        self.num_sites = len(T)
        self.t_max = max(T)
        self.slot_words = SLOT_HEADER_WORDS + 2 * self.t_max
        self._scratch_offsets = [0] * len(T)
        for i in range(1, len(T)):
            self._scratch_offsets[i] = self._scratch_offsets[i - 1] + 4 * T[i - 1]
        self.replays_per_readback = int(replays_per_readback)
        self.capture_wait_timeout = float(capture_wait_timeout)
        self.overflow_check = bool(overflow_check)
        self.on_call = on_call

        i64 = dict(dtype=torch.int64, device=self.device)
        with torch.cuda.device(self.device):
            self.cursor = torch.zeros(1, **i64)
            self.site_counters = torch.zeros(self.num_sites, **i64)
            self.scratch = torch.zeros(max(1, 4 * sum(T)), dtype=torch.int32, device=self.device)
            self.enable = torch.ones(self.num_sites, dtype=torch.int32, device=self.device)
            self.replay_epoch = torch.zeros(1, **i64)
            self.ring_desc = torch.zeros(2, **i64)
            self._copy_stream = torch.cuda.Stream(self.device)

        self._cond = threading.Condition(threading.RLock())
        self._tls = threading.local()
        self._active_captures = 0
        self._graph_calls: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self._next_call_id = 1
        self._host_epoch = 0
        self._blocks: list[Optional[_Block]] = [None, None]
        self._cur = 0
        self._issued = 0  # calls issued into the current ring since it became current
        self._pending: "collections.deque[ReadbackHandle]" = collections.deque()
        self._last = 0
        self._expected_k = np.zeros(self.num_sites, dtype=np.int64)
        self._unreported: list[ReadbackResult] = []
        self._closed = False

        _op.register(self)
        try:
            with torch.cuda.device(self.device):
                if capacity:
                    self._blocks = [self._new_block(int(capacity)), self._new_block(int(capacity))]
                    self.ring_desc.copy_(self._blocks[0].desc)
                # State must be initialised before any stream (or capture) uses it.
                torch.cuda.current_stream(self.device).synchronize()
        except BaseException:
            _op.unregister(self)
            raise

    # ----------------------------------------------------------- public API

    def probe(self, tensors: Sequence[torch.Tensor], site: int) -> None:
        """Record one call of ``site`` fingerprinting ``tensors`` (via ``opguard::probe``)."""
        torch.ops.opguard.probe(list(tensors), int(site), self.ring_desc, self.cursor,
                                self.site_counters, self.scratch, self.enable, self.replay_epoch)

    @contextlib.contextmanager
    def capture(self, graph: torch.cuda.CUDAGraph, **graph_kwargs):
        """Capture ``graph`` (via ``torch.cuda.graph``) and bind its probe count to it.

        Use ``replay`` to replay it so the recorder can reserve ring space first.
        While any thread is inside ``capture``, other threads never issue CUDA
        calls that could invalidate the capture (synchronisation, event
        queries, allocation): they wait, or raise for explicit readbacks.
        """
        tls = self._tls
        if getattr(tls, "in_capture", False):
            raise RuntimeError("nested rec.capture() on the same thread is not supported")
        with self._cond:
            self._check_open()
            self._active_captures += 1
            tls.in_capture, tls.count = True, 0
        ok = False
        try:
            with torch.cuda.graph(graph, **graph_kwargs):
                yield graph
            ok = True
        finally:
            with self._cond:
                tls.in_capture = False
                self._active_captures -= 1
                if ok:
                    self._graph_calls[graph] = tls.count  # a re-capture replaces the count
                else:
                    self._graph_calls.pop(graph, None)
                self._cond.notify_all()

    def replay(self, graph: torch.cuda.CUDAGraph) -> int:
        """Reserve space, bump the replay epoch on the current stream, replay; return the epoch."""
        if getattr(self._tls, "in_capture", False) or self._stream_capturing():
            raise RuntimeError("rec.replay() cannot be called while capturing")
        with self._cond:
            self._check_open()
            calls = self._graph_calls.get(graph)
            if calls is None:
                raise RuntimeError("graph was not captured with rec.capture(); cannot reserve space for it")
            with torch.cuda.device(self.device):
                self._ensure_space(calls, for_replay=True)
                self.replay_epoch.add_(1)
                graph.replay()
            self._host_epoch += 1
            return self._host_epoch

    def set_enabled(self, site: int, enabled: bool) -> None:
        """Switch ``site`` on or off with a write on the current stream (stream-ordered with probes)."""
        self._check_site(site)
        if self._stream_capturing():
            raise RuntimeError("set_enabled() would be captured into the graph; call it outside capture")
        with self._cond, torch.cuda.device(self.device):
            self.enable[site:site + 1].fill_(1 if enabled else 0)

    def readback_async(self, join_streams: Sequence[torch.cuda.Stream] = ()) -> ReadbackHandle:
        """Switch rings and issue the readback of the old one; collect it later with ``collect``.

        Must be called at a safe point (see the module docstring); streams in
        ``join_streams`` are joined into the current stream first.
        """
        with self._cond:
            self._check_explicit_readback()
            if self._blocks[self._cur] is None:
                return ReadbackHandle(None)
            with torch.cuda.device(self.device):
                return self._switch(0, False, join_streams)

    def collect(self, handle: ReadbackHandle) -> ReadbackResult:
        """Wait for ``handle`` (and earlier readbacks); return everything collected since last reported."""
        with self._cond:
            self._check_explicit_readback()
            self._collect_through(handle)
            return self._drain()

    def readback(self, join_streams: Sequence[torch.cuda.Stream] = ()) -> ReadbackResult:
        """Blocking readback; returns all results collected since the last report (automatic ones too)."""
        with self._cond:
            return self.collect(self.readback_async(join_streams))

    def close(self) -> None:
        """Finish pending copies, release the rings and unregister from the device."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            for handle in self._pending:
                handle.event.synchronize()
                handle.done = True
            self._pending.clear()
            for block in self._blocks:
                if block is not None:
                    self._free_block(block)
            self._blocks = [None, None]
        _op.unregister(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -------------------------------------------------------- probe body

    def _probe_impl(self, tensors, site, ring_desc, cursor, site_counters, scratch, enable, replay_epoch):
        """Python body of ``opguard::probe``; kernels use the tensors passed in."""
        self._check_site(site)
        if len(tensors) > self.tensors_per_site[site]:
            raise ValueError(f"opguard probe site {site}: {len(tensors)} tensors exceed its limit "
                             f"of {self.tensors_per_site[site]}")
        for t in tensors:
            if not t.is_cuda or t.device != self.device:
                raise ValueError(f"opguard probe site {site}: tensors must be CUDA tensors on {self.device}, "
                                 f"got one on {t.device}")
        tls = self._tls
        with torch.cuda.device(self.device):
            capturing = torch.cuda.is_current_stream_capturing()
            in_capture = getattr(tls, "in_capture", False)
            if capturing and not in_capture:
                raise RuntimeError(f"opguard probe site {site}: the current stream is being captured outside "
                                   "rec.capture(); space for its replays could not be reserved")
            if in_capture and not capturing:
                raise RuntimeError(f"opguard probe site {site}: called on a non-capturing stream inside "
                                   "rec.capture()")
            # (1) allocate the call id under the lock, (2) run the callback without it
            # (it may call probe again), (3) reserve space and launch atomically.
            with self._cond:
                self._check_open()
                call_id = self._next_call_id
                self._next_call_id += 1
            callback = self.on_call
            if callback is not None:
                callback(site, call_id, tensors, capturing)
            sources = [t.detach().resolve_conj().resolve_neg().contiguous() for t in tensors]
            with self._cond:
                if capturing:
                    tls.count += 1
                else:
                    self._ensure_space(1, for_replay=False)
                self._ext.device_record_probe(
                    sources, site, self._scratch_offsets[site], self.tensors_per_site[site], call_id,
                    self.slot_words, ring_desc, cursor, site_counters, scratch, enable, replay_epoch)

    # ---------------------------------------------------- space reservation

    def _needs_space(self, n: int) -> bool:
        block = self._blocks[self._cur]
        if block is None:
            return True
        return self.overflow_check and self._issued + n > block.cap

    def _ensure_space(self, n: int, for_replay: bool) -> None:
        """Make room for ``n`` calls about to be issued. Lock held."""
        while self._needs_space(n):
            self._wait_for_captures()  # may release the lock; re-check afterwards
            if not self._needs_space(n):
                break
            self._switch(n, for_replay, ())
        self._issued += n

    def _switch(self, n: int, for_replay: bool, join_streams) -> ReadbackHandle:
        """Continue in the other ring (grown if it cannot hold ``n``) and read the current one back."""
        cur = self._blocks[self._cur]
        other_index = 1 - self._cur
        other = self._blocks[other_index]
        if other is not None and other.pending is not None:
            self._collect_through(other.pending)
        if other is None or other.cap < max(n, 1):
            target = cur.cap if cur is not None else 0
            if target < max(n, 1):
                target = self._grow_cap(n, for_replay, target)
            if other is not None:
                self._free_block(other)
            other = self._blocks[other_index] = self._new_block(target)
        if cur is None:
            # No ring existed, so no commit can have written one: just install it.
            self.ring_desc.copy_(other.desc)
            handle = ReadbackHandle(None)
        else:
            handle = self._issue_readback(cur, other, join_streams)
        self._cur = other_index
        self._issued = 0
        return handle

    def _grow_cap(self, n: int, for_replay: bool, old: int) -> int:
        if for_replay:
            return max(n * self.replays_per_readback, n, 1)
        return max(n, 2 * old, MIN_EAGER_CAPACITY)

    def _new_block(self, cap: int) -> _Block:
        return _Block(cap, self.slot_words, self.num_sites, self.device)

    def _free_block(self, block: _Block) -> None:
        block.ring.record_stream(self._copy_stream)
        block.snap.record_stream(self._copy_stream)
        block.ring = block.desc = block.snap = block.host = block.snap_host = block.c1_host = None

    def _wait_for_captures(self) -> None:
        """Wait (lock released, no CUDA calls) until no capture of this recorder is in progress."""
        if self._active_captures == 0:
            return
        if getattr(self._tls, "in_capture", False):
            raise RuntimeError("recorder space cannot be reserved from inside rec.capture()")
        deadline = time.monotonic() + self.capture_wait_timeout
        while self._active_captures > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"opguard device recorder: waited {self.capture_wait_timeout:g}s for another thread's "
                    "CUDA graph capture on this recorder to finish; it never did. The capturing thread may "
                    "itself be waiting on this thread (mutual wait).")
            self._cond.wait(remaining)

    # ------------------------------------------------------------- readback

    def _issue_readback(self, block: _Block, next_block: _Block, join_streams) -> ReadbackHandle:
        """Snap protocol steps 1-5 for ``block``, switching to ``next_block``. Lock held, no capture."""
        assert block.pending is None  # its previous readback was collected before switching back
        stream = torch.cuda.current_stream(self.device)
        for side in join_streams:
            joined = torch.cuda.Event()
            joined.record(side)
            stream.wait_event(joined)
        block.snap[:1].copy_(self.cursor)
        block.snap[1:].copy_(self.site_counters)
        self.ring_desc.copy_(next_block.desc)  # commits after the event write next_block
        snapped = torch.cuda.Event()
        snapped.record(stream)
        copy = self._copy_stream
        with torch.cuda.stream(copy):
            copy.wait_event(snapped)
            block.snap_host.copy_(block.snap, non_blocking=True)
            block.host.copy_(block.ring, non_blocking=True)
            # Statistics only: c1 - c0 is how many commits (into next_block) ran
            # while this ring was copied. It plays no part in accepting slots.
            block.c1_host.copy_(self.cursor, non_blocking=True)
            done = torch.cuda.Event()
            done.record(copy)
        handle = block.pending = ReadbackHandle(block, done)
        self._pending.append(handle)
        return handle

    def _collect_through(self, handle: ReadbackHandle) -> None:
        """Wait for and parse pending readbacks in issue order, up to ``handle``."""
        while not handle.done:
            head = self._pending.popleft()
            head.event.synchronize()
            self._unreported.append(self._parse(head.block))
            head.done = True
            head.block.pending = None

    def _parse(self, block: _Block) -> ReadbackResult:
        """Snap protocol steps 6-7 plus loss detection, vectorized with numpy."""
        snap = block.snap_host.numpy().copy()
        c0, counts = int(snap[0]), snap[1:]
        c1 = int(block.c1_host.numpy()[0])
        last = self._last
        if c0 < last or (counts < self._expected_k).any():
            raise RuntimeError("opguard device recorder: counters went backwards (internal error)")
        seqs = np.arange(last, c0, dtype=np.int64)
        rows = block.host.numpy()[seqs % block.cap]
        ok = rows[:, 0] == seqs  # the ring is not written after the snapshot
        rows, seqs = rows[ok], seqs[ok]
        site, n, k = rows[:, 1].copy(), rows[:, 2].copy(), rows[:, 3].copy()
        if site.size:
            if site.min() < 0 or site.max() >= self.num_sites:
                raise RuntimeError("opguard device recorder: accepted a slot with an invalid site (internal error)")
            limits = np.asarray(self.tensors_per_site, dtype=np.int64)[site]
            if (n < 0).any() or (n > limits).any():
                raise RuntimeError("opguard device recorder: accepted a slot with an invalid tensor count (internal error)")
        fingerprints = np.ascontiguousarray(rows[:, SLOT_HEADER_WORDS:]).view(np.uint32)
        fingerprints = fingerprints.reshape(len(seqs), self.t_max, 4)
        fingerprints[np.arange(self.t_max)[None, :] >= n[:, None]] = 0

        ph_site, ph_k = self._placeholders(site, k, counts)
        self._last = c0
        return ReadbackResult(seqs, site, n, k, rows[:, 4].copy(), rows[:, 5].copy(), fingerprints,
                              ph_site, ph_k, lost=int(c0 - last - len(seqs)),
                              written_during_copy=c1 - c0, readbacks=1)

    def _placeholders(self, site, k, counts):
        """Per-site gaps between the expected k, the accepted records and the snapshot count."""
        order = np.lexsort((k, site))
        site, k = site[order], k[order]
        expected = self._expected_k
        active = np.nonzero(counts > expected)[0]
        if site.size and not np.isin(np.unique(site), active).all():
            raise RuntimeError("opguard device recorder: record beyond the snapshot count; "
                               "was readback called outside a safe point?")
        bounds = np.searchsorted(site, active, side="left"), np.searchsorted(site, active, side="right")
        out_site, out_k = [], []
        for i, lo, hi in zip(active, *bounds):
            ks, e, total = k[lo:hi], expected[i], counts[i]
            if ks.size:
                if ks[0] < e or (np.diff(ks) <= 0).any():
                    raise RuntimeError(f"opguard device recorder: duplicate record for site {i} (internal error)")
                if ks[-1] >= total:
                    raise RuntimeError("opguard device recorder: record beyond the snapshot count; "
                                       "was readback called outside a safe point?")
            missing = np.setdiff1d(np.arange(e, total, dtype=np.int64), ks, assume_unique=True)
            if missing.size:
                out_site.append(np.full(missing.size, i, dtype=np.int64))
                out_k.append(missing)
            expected[i] = total
        if not out_site:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy()
        return np.concatenate(out_site), np.concatenate(out_k)

    def _drain(self) -> ReadbackResult:
        results, self._unreported = self._unreported, []
        return ReadbackResult.concat(results, self.t_max)

    # ------------------------------------------------------------- checks

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeviceRecorder is closed")

    def _check_site(self, site: int) -> None:
        if not 0 <= site < self.num_sites:
            raise ValueError(f"opguard probe site {site}: out of range (recorder has {self.num_sites} sites)")

    def _stream_capturing(self) -> bool:
        with torch.cuda.device(self.device):
            return torch.cuda.is_current_stream_capturing()

    def _check_explicit_readback(self) -> None:
        self._check_open()
        if self._active_captures or self._stream_capturing():
            raise RuntimeError("readback is not allowed while a CUDA graph capture on this recorder is in "
                               "progress (it would synchronize and break the capture)")
