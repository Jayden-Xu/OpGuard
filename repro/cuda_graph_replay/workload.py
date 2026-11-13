"""One traced op, run eagerly (GRAPH=0) or through a CUDA graph (GRAPH=1).

Both modes execute the same computation on the same inputs: step 0 eagerly,
then steps 1-3 either eagerly or as graph replays.
"""
import os

import torch


def step(x, w):
    out = torch.add(x, w)
    return out


gen = torch.Generator(device="cuda").manual_seed(0)
inputs = [torch.randn(1024, device="cuda", generator=gen) for _ in range(4)]
x = torch.empty(1024, device="cuda")  # static input buffer
w = torch.ones(1024, device="cuda")

x.copy_(inputs[0])
step(x, w)  # step 0: eager in both modes

if os.environ["GRAPH"] == "1":
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(x, w)  # captured, not executed
    for i in (1, 2, 3):
        x.copy_(inputs[i])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, inputs[i] + w)  # the computation itself is correct
else:
    for i in (1, 2, 3):
        x.copy_(inputs[i])
        step(x, w)
torch.cuda.synchronize()
