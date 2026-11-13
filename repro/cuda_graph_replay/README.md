# CUDA graph replay 拿不到正确的 trace 记录

即使按官方流程走（discover 用 eager，插桩后再开 CUDA graph 复跑），graph replay 也没有正确的指纹记录。

## 运行

需要 CUDA GPU，opguard 可导入（已安装，或 `PYTHONPATH=<OpGuard checkout>`）。不需要 native 扩展：XOR 指纹会走 PyTorch fallback，discover 只开 python 后端。

```bash
python repro/cuda_graph_replay/repro.py
```

`repro.py` 把 `workload.py` 拷到临时目录，依次调用官方 backend（`python -m opguard.tools`）：

1. `discover`（eager）
2. `prepare`
3. `trace`：eager 跑一次，作为参考答案
4. `trace`：开 CUDA graph 再跑一次

## 结果（main 070931b，RTX 3090，torch 2.13.0+cu130）

```
eager trace: 4 records for `out = torch.add(x, w)` (steps 0-3)
graph trace: 2 records (expected 4: step 0 + replays 1-3)
  graph record 0, made at step 0 (eager)  : input and output fingerprints equal eager step 0
  graph record 1, made at capture         : input and output fingerprints equal eager step 3
```

replay 的计算本身是对的，`workload.py` 里的 assert 会在每次 replay 后检查。

## 原因

- **replay 没有记录。** replay 不经过 Python，插桩的 wrapper 根本不会执行。
- **capture 记录的值不属于 capture 那次调用。** capture 时 wrapper 照常执行，指纹 kernel 被录进 graph，每次 replay 都会把本次的指纹写进同一块 graph 内存池 buffer。记录里只存了这块 buffer 的引用，dump 线程什么时候读，就得到什么值：
  - 默认在进程退出时读，这时是最后一次 replay（step 3）的值；
  - 如果在第一次 replay 之前读，得到的是全 0 或内存池里的旧数据。

  capture 那次调用本身并没有执行，所以这条记录的值永远不是它的。
- **后果：对比报告会配错。** `compare` 会把这条记录按输入指纹配到 eager 的 step 3，显示 `ok`；step 1–2 显示 `unaligned`。
