# OP02 — Core 注意力边界计算

独立包名：`vllm-hust-op02-core-attention-boundary`

GitHub 仓库：`xmdhb/vllm-hust-core-attention-boundary`

对应 Core #41：用累计计数和 `searchsorted` 计算 decode/extend/prefill 边界。

启用变量：

```bash
export VLLM_HUST_OP02_CORE_ATTENTION_BOUNDARY_ENABLE=1
export VLLM_HUST_OP02_CORE_ATTENTION_BOUNDARY_EVIDENCE=1
```

默认关闭，`VLLM_HUST_OP02_CORE_ATTENTION_BOUNDARY_KILL_SWITCH=1` 优先级最高。

打包：在本目录执行 `python3 -m build --wheel`。测试：`PYTHONPATH=src python3 -m pytest -q tests`。
