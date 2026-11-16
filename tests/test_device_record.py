"""Tests for the device-side recorder (``opguard.device_record``).

Every test needs a CUDA device. Reference fingerprints come from the upstream CPU
path ``xor_signature(t.cpu(), regional=4, by="words")``.
"""
import functools
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")

from opguard.device_record import DeviceRecorder  # noqa: E402
from opguard.tracer.xor_signature import xor_signature  # noqa: E402

DEV = torch.device("cuda", 0)


def isolated(test):
    """Run ``test`` in a fresh interpreter.

    These tests need real GPU concurrency. Other test modules can turn on CUPTI
    kernel activity tracing for the whole process (some do so at import time,
    i.e. during collection), and that serializes every kernel, so no overlap
    could ever be observed in a shared pytest process.
    """
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        if os.environ.get("OPGUARD_DEVICE_RECORD_CHILD") == test.__name__:
            return test(*args, **kwargs)
        env = dict(os.environ, OPGUARD_DEVICE_RECORD_CHILD=test.__name__)
        cmd = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
               f"{os.path.abspath(__file__)}::{test.__name__}"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        print(proc.stdout[-4000:])
        assert proc.returncode == 0, proc.stdout[-8000:] + proc.stderr[-4000:]
        assert " passed" in proc.stdout and "skipped" not in proc.stdout, proc.stdout[-4000:]

    return wrapper


def ref(t):
    return xor_signature(t.detach().cpu(), regional=4, by="words").numpy().astype(np.uint32)


@pytest.fixture(scope="module", autouse=True)
def _warm_kernels():
    """Load every kernel once so later captures never load modules lazily."""
    if not torch.cuda.is_available():
        yield
        return
    rec = DeviceRecorder(DEV, [4], capacity=8)
    raw = torch.arange(64, dtype=torch.uint8, device=DEV)
    rec.probe([raw, raw[1:], torch.randn(3000, device=DEV)], 0)
    rec.probe([], 0)
    torch.randn(8, 8, device=DEV) @ torch.randn(8, 8, device=DEV)
    rec.readback()
    rec.close()
    torch.cuda.synchronize()
    yield


@pytest.fixture
def make_rec():
    made = []

    def _make(sites, **kw):
        rec = DeviceRecorder(kw.pop("device", DEV), sites, **kw)
        made.append(rec)
        return rec

    yield _make
    torch.cuda.synchronize()
    for rec in made:
        rec.close()


def assert_records(res, expected, key=lambda res, i: int(res.call_id[i])):
    """Every accepted record matches ``expected[key]`` (list of 4-word refs)."""
    for i in range(len(res)):
        exp = expected[key(res, i)]
        assert int(res.n[i]) == len(exp), (i, int(res.n[i]), len(exp))
        for j, e in enumerate(exp):
            np.testing.assert_array_equal(res.fingerprints[i, j], e, err_msg=f"record {i} tensor {j}")
        assert not res.fingerprints[i, len(exp):].any()


class EagerLog:
    """Probe eagerly and remember the reference value at that point in the stream."""

    def __init__(self, rec):
        self.rec, self.expected, self.ids = rec, {}, []
        rec.on_call = lambda site, cid, tensors, capturing: self.ids.append(cid)

    def probe(self, tensors, site):
        before = len(self.ids)
        self.rec.probe(tensors, site)
        assert len(self.ids) == before + 1
        self.expected[self.ids[-1]] = [ref(t) for t in tensors]


# ----------------------------------------------------------------- model helpers

class Model:
    """matmul, add, gelu, layernorm and an in-place op; seven probes per forward."""

    PROBES = 7

    def __init__(self, dev=DEV, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        self.x = torch.randn(64, 128, device=dev, generator=g)
        self.w = torch.randn(128, 128, device=dev, generator=g) / 11
        self.b = torch.randn(128, device=dev, generator=g)

    def forward(self, probe, base_site=0):
        """Return, in probe order, the tensors whose values each probe must record."""
        s = base_site
        y = self.x @ self.w
        probe([y], s + 0)
        z = y + self.b
        probe([z], s + 1)
        g = F.gelu(z)
        probe([g], s + 2)
        ln = F.layer_norm(g, (128,))
        probe([ln], s + 3)
        before = ln.clone()  # value seen by probe s+3
        ln.mul_(2.0)
        probe([ln], s + 4)
        probe([ln, z], s + 5)
        probe([], s + 6)
        return [[y], [z], [g], [before], [ln], [ln, z], []]

    def new_input(self):
        self.x.copy_(torch.randn_like(self.x))


MODEL_T = [1, 1, 1, 1, 1, 2, 0]


def capture(rec, build):
    """Capture ``build(rec.probe)`` with ``rec.capture``; map call id -> probed tensors."""
    ids, prev = [], rec.on_call
    rec.on_call = lambda site, cid, tensors, capturing: ids.append((cid, capturing))
    graph = torch.cuda.CUDAGraph()
    with rec.capture(graph):
        probed = build(rec.probe)
    rec.on_call = prev
    assert all(c for _, c in ids) and len(ids) == len(probed)
    return graph, dict(zip([cid for cid, _ in ids], probed))


def replay_and_expect(rec, graph, probed, expected, before=None):
    if before is not None:
        before()
    epoch = rec.replay(graph)
    torch.cuda.synchronize()
    for cid, tensors in probed.items():
        expected[(epoch, cid)] = [ref(t) for t in tensors]
    return epoch


def by_epoch_call(res, i):
    return int(res.replay_epoch[i]), int(res.call_id[i])


def warm_model(model, times=2):
    for _ in range(times):
        model.forward(lambda tensors, site: None)
    torch.cuda.synchronize()


# ------------------------------------------------------------------ 1. eager


def test_eager_fingerprints_match_reference(make_rec):
    rec = make_rec([3, 3])
    log = EagerLog(rec)
    u8 = torch.randint(0, 256, (4099,), dtype=torch.uint8, device=DEV)
    cases = [
        torch.randn(1000, device=DEV).to(torch.bfloat16),
        torch.randn(37, 13, device=DEV),
        torch.randint(-2**62, 2**62, (513,), device=DEV),
        torch.randint(0, 256, (1001,), dtype=torch.uint8, device=DEV),
        torch.rand(333, device=DEV) > 0.5,
        torch.randint(0, 256, (7,), dtype=torch.uint8, device=DEV),
        u8[1:], u8[2:], u8[3:], u8[1:-2], u8[3:6],
        torch.randn(64, 33, device=DEV).t(),
        torch.empty(0, device=DEV),
        torch.tensor(3.5, device=DEV),
        torch.randn(50, dtype=torch.cfloat, device=DEV).conj(),
        torch._neg_view(torch.randn(50, dtype=torch.cfloat, device=DEV)),
        torch._neg_view(torch.randn(41, device=DEV)),
        torch.randn(4 * 1024 * 1024, device=DEV),                       # 16 MB
        torch.randint(0, 256, (16 * 1024 * 1024 + 3,), dtype=torch.uint8, device=DEV)[1:],
    ]
    assert u8[1:].storage_offset() == 1 and u8[3:].storage_offset() == 3
    for t in cases:
        log.probe([t], 0)
    # Several tensors in one call are fingerprinted independently.
    log.probe([cases[0], u8[3:], cases[12]], 1)
    # In-place updates: each probe sees the value at its stream position.
    x = torch.randn(1000, device=DEV)
    log.probe([x], 1)
    x.add_(1.0)
    log.probe([x], 1)
    x.mul_(3.0)
    log.probe([x], 1)

    res = rec.readback()
    assert len(res) == len(log.ids) and res.lost == 0 and len(res.placeholder_k) == 0
    assert sorted(res.call_id.tolist()) == sorted(log.ids)
    assert_records(res, log.expected)
    assert res.s.tolist() == list(range(len(log.ids)))


# ------------------------------------------------------- 2. empty tensor list


def test_empty_tensor_list_commits(make_rec):
    rec = make_rec([0, 1], capacity=16)
    rec.probe([], 0)
    rec.probe([torch.ones(3, device=DEV)], 1)
    graph, _ = capture(rec, lambda probe: probe([], 0) or [[]])
    for _ in range(3):
        rec.replay(graph)
    res = rec.readback()
    assert res.lost == 0 and len(res) == 5
    assert res.s.tolist() == [0, 1, 2, 3, 4]
    assert res.n.tolist() == [0, 1, 0, 0, 0]
    assert res.site.tolist() == [0, 1, 0, 0, 0]
    assert res.k.tolist() == [0, 0, 1, 2, 3]
    assert res.replay_epoch.tolist()[2:] == [1, 2, 3]


# ---------------------------------------------- 3/4/5. graph, call ids, epochs


def test_graph_replay_records_every_probe(make_rec):
    rec = make_rec(MODEL_T, capacity=64)
    model = Model()
    warm_model(model)
    graph, probed = capture(rec, model.forward)
    captured = rec.readback()
    assert len(captured) == 0 and len(captured.placeholder_k) == 0  # capture writes nothing

    expected, epochs = {}, []
    for _ in range(50):
        epochs.append(replay_and_expect(rec, graph, probed, expected, model.new_input))
    res = rec.readback()
    assert epochs == list(range(1, 51))
    assert len(res) == 50 * Model.PROBES and res.lost == 0 and len(res.placeholder_k) == 0
    assert_records(res, expected, by_epoch_call)
    # Call ids: replay records carry the ids handed out during capture.
    assert set(res.call_id.tolist()) == set(probed)
    # Replay epochs: each record carries the epoch returned by rec.replay.
    for epoch in epochs:
        sel = res.replay_epoch == epoch
        assert sel.sum() == Model.PROBES
        assert sorted(res.call_id[sel].tolist()) == sorted(probed)
    # Each replay produces one record per probe, in program order.
    order = res.site.reshape(50, Model.PROBES)
    assert (order == np.arange(Model.PROBES)).all()
    assert (res.k.reshape(50, Model.PROBES) == np.arange(50)[:, None]).all()


def test_graph_errors(make_rec):
    rec = make_rec(MODEL_T, capacity=64)
    model = Model()
    warm_model(model)
    foreign = torch.cuda.CUDAGraph()
    with torch.cuda.graph(foreign):
        model.forward(lambda tensors, site: None)
    with pytest.raises(RuntimeError, match="not captured"):
        rec.replay(foreign)

    other = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="rec.capture|capture"):
        with torch.cuda.graph(other):
            rec.probe([model.x], 0)
    res = rec.readback()
    assert len(res) == 0 and res.lost == 0


def test_eager_call_ids_unique(make_rec):
    rec = make_rec([1, 1], capacity=8)
    log = EagerLog(rec)
    t = torch.randn(10, device=DEV)
    for i in range(40):
        log.probe([t], i % 2)
    res = rec.readback()
    assert len(set(log.ids)) == 40
    assert res.call_id.tolist() == log.ids  # one record per call, same order
    assert_records(res, log.expected)


# ---------------------------------------------------------------- 6. enable


def test_enable_toggle_without_recapture(make_rec):
    rec = make_rec(MODEL_T, capacity=64)
    model = Model()
    warm_model(model)
    graph, probed = capture(rec, model.forward)
    expected = {}
    for _ in range(3):
        replay_and_expect(rec, graph, probed, expected, model.new_input)
    rec.set_enabled(2, False)
    rec.set_enabled(5, False)
    off_epochs = [replay_and_expect(rec, graph, probed, expected, model.new_input) for _ in range(4)]
    rec.set_enabled(2, True)
    rec.set_enabled(5, True)
    for _ in range(3):
        replay_and_expect(rec, graph, probed, expected, model.new_input)

    res = rec.readback()
    assert res.lost == 0 and len(res.placeholder_k) == 0
    assert len(res) == 10 * Model.PROBES - 4 * 2
    assert_records(res, expected, by_epoch_call)
    off = np.isin(res.replay_epoch, off_epochs)
    assert not np.isin(res.site[off], [2, 5]).any()
    # Disabled sites take neither a sequence number nor a per-site number.
    assert res.s.tolist() == list(range(len(res)))
    for site in (2, 5):
        assert res.k[res.site == site].tolist() == list(range(6))
    for site in (0, 1, 3, 4, 6):
        assert res.k[res.site == site].tolist() == list(range(10))


# ------------------------------------------------------------ 7. too many


def test_too_many_tensors_names_site(make_rec):
    rec = make_rec([1, 2, 1], capacity=8)
    t = torch.ones(4, device=DEV)
    with pytest.raises(ValueError, match="site 1"):
        rec.probe([t, t, t], 1)
    with pytest.raises(ValueError, match="site 3"):
        rec.probe([t], 3)
    with pytest.raises(ValueError, match="site 0"):
        rec.probe([t.cpu()], 0)
    res = rec.readback()
    assert len(res) == 0 and res.lost == 0  # nothing half-issued


# ------------------------------------------------------ 8. lazy allocation


def test_late_allocation_and_growth(make_rec):
    rec = make_rec(MODEL_T, capacity=0)
    model = Model()
    warm_model(model)
    graph_a, probed_a = capture(rec, model.forward)
    assert rec.ring_desc.tolist() == [0, 0]
    expected = {}
    replay_and_expect(rec, graph_a, probed_a, expected, model.new_input)
    assert rec.ring_desc[1].item() == 7  # allocated on the first replay, sized for it
    replay_and_expect(rec, graph_a, probed_a, expected, model.new_input)

    graph_b, probed_b = capture(rec, lambda probe: sum((model.forward(probe) for _ in range(5)), []))
    assert len(probed_b) == 35
    for _ in range(3):
        replay_and_expect(rec, graph_b, probed_b, expected, model.new_input)
    for _ in range(4):  # graph A must now write into the new, larger block
        replay_and_expect(rec, graph_a, probed_a, expected, model.new_input)
    res = rec.readback()
    assert res.lost == 0 and len(res.placeholder_k) == 0
    assert len(res) == 6 * 7 + 3 * 35
    assert_records(res, expected, by_epoch_call)
    assert res.s.tolist() == list(range(len(res)))


# ------------------------------------------------------ 9. overflow fallback


def test_overflow_fallback(make_rec):
    rec = make_rec(MODEL_T, capacity=16, overflow_check=False)
    model = Model()
    warm_model(model)
    graph, probed = capture(rec, lambda probe: sum((model.forward(probe) for _ in range(5)), []))
    expected = {}
    for i in range(6):
        replay_and_expect(rec, graph, probed, expected, model.new_input)
        res = rec.readback()
        assert len(res) == 16
        assert res.s.tolist() == list(range(35 * i + 19, 35 * (i + 1)))
        assert res.lost == 19 == len(res.placeholder_k)
        assert_records(res, expected, by_epoch_call)
        # Placeholders and records together cover every call of this replay.
        assert len(res.placeholder_k) + len(res) == 35


# ---------------------------------------------- 10. concurrent copy and write


@isolated
def test_snap_readback_under_concurrent_writes(make_rec):
    P, C, replays_per_cycle = 32, 0x0123456789ABCDEF, 3
    rec = make_rec([1] * P, capacity=2 * replays_per_cycle * P)
    sites = torch.arange(P, device=DEV)
    const = torch.full((P,), C, dtype=torch.int64, device=DEV)

    def build(probe):
        vals = torch.stack([rec.replay_epoch * 1000 + sites, const], dim=1)
        for i in range(P):
            probe([vals[i]], i)
        return [[vals[i]] for i in range(P)]

    graph, probed = capture(rec, build)
    ids = np.array(list(probed), dtype=np.int64)  # call id at each position
    lo32 = lambda a: (a & 0xFFFFFFFF).astype(np.uint32)
    hi32 = lambda a: ((a >> 32) & 0xFFFFFFFF).astype(np.uint32)

    def wrong(res):
        """Records whose content does not belong to their sequence number."""
        epoch, site, fp = res.replay_epoch, res.site, res.fingerprints[:, 0, :]
        exp_epoch = res.s // P + 1
        exp_site = res.s % P
        v = epoch * 1000 + site
        bad = (epoch != exp_epoch) | (site != exp_site) | (res.k != exp_epoch - 1)
        bad |= res.call_id != ids[exp_site]
        bad |= (fp[:, 0] != lo32(v)) | (fp[:, 1] != hi32(v))
        bad |= (fp[:, 2] != np.uint32(C & 0xFFFFFFFF)) | (fp[:, 3] != np.uint32(C >> 32))
        return int(bad.sum())

    total = accepted = lost = false_accept = readbacks = during = max_during = 0
    while total < 1_000_000:
        # Replays issued before the readback keep the GPU busy, so the replays issued
        # after it are already queued when the snapshot point is reached and their
        # commits run while the copy stream copies the ring that was switched out.
        for _ in range(replays_per_cycle):
            rec.replay(graph)
        handle = rec.readback_async()
        for _ in range(replays_per_cycle):
            rec.replay(graph)
        total += 2 * replays_per_cycle * P
        res = rec.collect(handle)
        accepted, lost, false_accept = accepted + len(res), lost + res.lost, false_accept + wrong(res)
        readbacks += res.readbacks
        during += res.written_during_copy
        max_during = max(max_during, res.written_during_copy)
    res = rec.readback()
    accepted, lost, false_accept = accepted + len(res), lost + res.lost, false_accept + wrong(res)
    readbacks += res.readbacks
    print(f"\n[snap] calls={total} readbacks={readbacks} accepted={accepted} lost={lost} "
          f"false_accept={false_accept} written_during_copy={during} max_per_readback={max_during}")
    assert accepted == total and lost == 0
    assert false_accept == 0
    assert during > 0  # commits really ran while rings were being copied


# --------------------------------------------------------- 11. multi stream


@isolated
def test_two_sites_on_parallel_streams(make_rec):
    rec = make_rec([1, 1], capacity=64)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    a = torch.randn(48 * 1024 * 1024, device=DEV)
    b = torch.randn(48 * 1024 * 1024, device=DEV)
    ra, rb = ref(a), ref(b)
    main = torch.cuda.current_stream()
    for _ in range(4):
        fork = torch.cuda.Event()
        fork.record(main)
        for st, t, site in ((s1, a, 0), (s2, b, 1)):
            with torch.cuda.stream(st):
                st.wait_event(fork)
                rec.probe([t], site)
        for st in (s1, s2):
            ev = torch.cuda.Event()
            ev.record(st)
            main.wait_event(ev)
    res = rec.readback(join_streams=(s1, s2))
    assert len(res) == 8 and res.lost == 0
    for i in range(len(res)):
        np.testing.assert_array_equal(res.fingerprints[i, 0], ra if res.site[i] == 0 else rb)


# ---------------------------------------------------------------- 12. opcheck


def test_opcheck(make_rec):
    rec = make_rec([2, 1], capacity=1024)
    t = torch.randn(17, 3, device=DEV)
    args = ([t, t[:, 1]], 0, rec.ring_desc, rec.cursor, rec.site_counters, rec.scratch,
            rec.enable, rec.replay_epoch)
    torch.library.opcheck(torch.ops.opguard.probe.default, args)
    torch.library.opcheck(torch.ops.opguard.probe.default,
                          ([], 1) + args[2:])


# ------------------------------------------------------- 13. threads


def test_two_threads_eager(make_rec):
    rec = make_rec([1, 1], capacity=32)
    ids, lock = [], threading.Lock()

    def on_call(site, cid, tensors, capturing):
        with lock:
            ids.append(cid)

    rec.on_call = on_call
    t = torch.randn(100, device=DEV)
    errors = []

    def worker(site):
        try:
            for _ in range(2000):
                rec.probe([t], site)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert not errors
    res = rec.readback()
    assert len(ids) == len(set(ids)) == 4000
    assert len(res) == 4000 and res.lost == 0 and len(res.placeholder_k) == 0
    assert sorted(res.call_id.tolist()) == sorted(ids)
    expected = ref(t)
    assert (res.fingerprints[:, 0] == expected).all()


def test_reentrant_on_call(make_rec):
    rec = make_rec([1, 1], capacity=8)
    t = torch.randn(9, device=DEV)
    seen = []

    def on_call(site, cid, tensors, capturing):
        seen.append((site, cid))
        if site == 0:
            rec.probe([t * 2], 1)

    rec.on_call = on_call
    rec.probe([t], 0)
    res = rec.readback()
    assert len(res) == 2 and res.lost == 0
    assert sorted(res.site.tolist()) == [0, 1]
    assert sorted(res.call_id.tolist()) == sorted(cid for _, cid in seen)


@pytest.mark.parametrize("capacity", [0, 4], ids=["allocate", "switch"])
def test_other_thread_waits_for_capture(make_rec, capacity):
    # capacity 0: B's eager probe must allocate the ring; capacity 4 (full): it
    # must switch rings and read back. Either could break A's capture.
    rec = make_rec(MODEL_T, capacity=capacity)
    model = Model()
    warm_model(model)
    side = torch.cuda.Stream()
    t = torch.randn(64, device=DEV)
    for _ in range(capacity):
        rec.probe([t], 0)
    torch.cuda.synchronize()
    assert rec.ring_desc[1].item() == capacity
    in_capture, b_done = threading.Event(), threading.Event()
    out = {}

    def thread_b():
        in_capture.wait()
        try:
            rec.readback()
        except RuntimeError as exc:
            out["readback_error"] = str(exc)
        start = time.monotonic()
        with torch.cuda.stream(side):
            rec.probe([t], 0)
        out["waited"] = time.monotonic() - start
        b_done.set()

    th = threading.Thread(target=thread_b)
    th.start()
    graph = torch.cuda.CUDAGraph()
    ids = []
    rec.on_call = lambda site, cid, tensors, capturing: ids.append((cid, capturing))
    with rec.capture(graph):
        probed = model.forward(rec.probe)
        in_capture.set()
        time.sleep(0.5)
        assert not b_done.is_set()  # B is blocked until the capture ends
    th.join(30)
    assert b_done.is_set()
    side.synchronize()
    assert "capture" in out["readback_error"]
    assert out["waited"] >= 0.3
    if capacity == 0:
        assert rec.ring_desc[1].item() > 0  # B allocated only after the capture
    rec.on_call = None
    probed = dict(zip([cid for cid, c in ids if c], probed))
    expected = {}
    for _ in range(5):
        replay_and_expect(rec, graph, probed, expected, model.new_input)
    res = rec.readback()
    assert res.lost == 0 and len(res.placeholder_k) == 0
    replay = res.replay_epoch > 0
    eager = ~np.isin(res.call_id, list(probed))
    assert eager.sum() == capacity + 1 and (replay & ~eager).sum() == 35
    assert_records(res[~eager], expected, by_epoch_call)
    assert (res.fingerprints[eager, 0] == ref(t)).all()


def test_capture_wait_timeout(make_rec):
    rec = make_rec([1], capacity=2, capture_wait_timeout=0.3)
    side = torch.cuda.Stream()
    t = torch.randn(64, device=DEV)
    rec.probe([t], 0)
    rec.probe([t], 0)
    torch.cuda.synchronize()
    in_capture = threading.Event()
    out = {}

    def thread_b():
        in_capture.wait()
        try:
            with torch.cuda.stream(side):
                rec.probe([t], 0)
        except RuntimeError as exc:
            out["error"] = str(exc)

    th = threading.Thread(target=thread_b)
    th.start()
    graph = torch.cuda.CUDAGraph()
    with rec.capture(graph):
        rec.probe([t], 0)
        in_capture.set()
        th.join(10)
    assert "another thread" in out["error"] and "wait" in out["error"]
    rec.replay(graph)
    res = rec.readback()
    assert res.lost == 0 and len(res) == 3


# ------------------------------------------------------------------ misc


def test_recapture_replaces_bound_count(make_rec):
    rec = make_rec([1], capacity=0)
    t = torch.randn(8, device=DEV)
    graph = torch.cuda.CUDAGraph()
    with rec.capture(graph):
        rec.probe([t], 0)
    rec.replay(graph)
    graph.reset()
    with rec.capture(graph):
        for _ in range(9):
            rec.probe([t], 0)
    for _ in range(3):
        rec.replay(graph)
    res = rec.readback()
    assert len(res) == 1 + 27 and res.lost == 0 and len(res.placeholder_k) == 0
