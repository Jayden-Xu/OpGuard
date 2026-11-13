"""CUDA graph replays get no correct trace records, even with the official flow.

Official backend flow on a copy of workload.py:
  discover (eager) -> prepare -> trace eager (reference) + trace with CUDA graph.
Eager records are correct, so they serve as the reference for the graph run.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from opguard.tools.operations import snapshot

work = Path(tempfile.mkdtemp(prefix="opguard-cuda-graph-"))
src = work / "src"  # instrumentation edits sources, so use a copy
src.mkdir()
shutil.copy(Path(__file__).with_name("workload.py"), src)


def backend(action, name, graph, **extra):
    request = {
        "protocol_version": 1, "output": str(work / name), "local_root": str(work / "local"),
        "timeout_seconds": 600, **extra,
        "workload": {"cwd": str(src), "script": str(src / "workload.py"), "args": [],
                     "launcher": [sys.executable], "ranks": [0],
                     "env": {"GRAPH": graph, "OPGUARD_CALLSITE_CUPTI": "0", "OPGUARD_CALLSITE_CUDAAPI": "0"}},
    }
    proc = subprocess.run([sys.executable, "-m", "opguard.tools", action],
                          input=json.dumps(request), capture_output=True, text=True)
    response = json.loads(proc.stdout.splitlines()[-1])
    if not response["ok"]:
        sys.exit(f"{action} failed: {response['error']}\n{proc.stderr[-3000:]}")
    return response["result"]


def add_records(capture):
    with snapshot(capture["output"]) as root:
        path = next(root.glob("exec_trace/gpu_000/ops/torch.add/*.jsonl"))
        records = [json.loads(line) for line in path.read_text().splitlines()]
    records.sort(key=lambda r: r["counter"])
    # args = (torch.add, x, w); fingerprints of input x and of the output
    return [(r["inputs"]["args"][1]["stats"]["xor_signature"], r["outputs"]["stats"]["xor_signature"])
            for r in records]


discovery = backend("discover", "discover", "0")
plan = backend("prepare", "plan", "0", discovery=discovery)
eager = add_records(backend("trace", "trace-eager", "0", plan=plan))
graph = add_records(backend("trace", "trace-graph", "1", plan=plan))

print(f"work dir: {work}")
print(f"eager trace: {len(eager)} records for `out = torch.add(x, w)` (steps 0-3)")
print(f"graph trace: {len(graph)} records (expected 4: step 0 + replays 1-3)")
for i, (record, made_at) in enumerate(zip(graph, ["step 0 (eager)", "capture"])):
    matches = [f"step {s}" for s, ref in enumerate(eager) if ref == record] or ["no eager step"]
    print(f"  graph record {i}, made at {made_at:16s}: input and output fingerprints equal eager {matches[0]}")
