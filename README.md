# OP02 — Core 注意力边界计算

公开包名：`vllm-hust-core-attention-boundary`
内部编号：`OP02`

GitHub 仓库：`xmdhb/vllm-hust-core-attention-boundary`

对应 Core #41：计算 decode/extend/prefill 边界；边界索引辅助函数在导入阶段不依赖 `torch`，运行时再使用张量完成边界和 token 数计算。

启用变量：

```bash
export VLLM_HUST_CORE_ATTENTION_BOUNDARY_ENABLE=1
export VLLM_HUST_CORE_ATTENTION_BOUNDARY_EVIDENCE=1
```

默认关闭，`VLLM_HUST_CORE_ATTENTION_BOUNDARY_KILL_SWITCH=1` 优先级最高。

打包：在本目录执行 `python3 -m build --wheel`。测试：`PYTHONPATH=src python3 -m pytest -q tests`。
