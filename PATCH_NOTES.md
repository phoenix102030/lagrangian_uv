# Stage-2 Lagrangian UV refactor notes

这个补丁主要处理三类问题：kernel 公式可辨识性、训练/内存效率、以及默认配置中会绕开 kernel 的 shortcut。

## 关键改动

1. `models/kernel.py`
   - 修正 same-component block：u->u 和 v->v 不再使用 `mu_i - mu_i = 0`。
   - same-component drift 使用 `dt * mu_i`，dispersion 使用 `I + 2 * dt^2 * Sigma_i`。
   - 默认关闭 u/v cross-component kernel block，只保留同一 wind component 的空间传播；需要时可设置 `model.allow_cross_component: true`。

2. `models/state_space.py`
   - 加入 `kernel_one_step_loss`，直接监督 `kernel_transition[t] @ observation[t-1] -> observation[t]`，避免 Kalman update、forcing、persistence 把 kernel 学习信号盖住。
   - validation/free-run rollout 不再使用 scheduled sampling。
   - 可选共享 u/v component 的 advection，默认开启 `shared_component_advection: true`，因为 advection 应该主要表示同一个二维运输速度场，而不是让 u/v 分别漂移。

3. `models/backbone.py`
   - 增加轻量 GRU temporal head，替代“每个时间点独立 CNN”的纯静态提取方式。
   - `forcing_scale <= 0` 时直接返回零 forcing，避免 NWP direct forcing shortcut。

4. `data.py`
   - 将经纬度转换为局地 km 坐标，并按 `local_km_scale` 缩放。否则 kernel 中的单位长度尺度和 lat/lon 度数不一致。

5. `train.py`
   - 支持 AMP。
   - DDP gradient accumulation 时使用 `no_sync()`，减少无效同步开销。
   - DataLoader 新增 `prefetch_factor`，默认减少 workers/pin_memory，避免 CPU/pinned memory 放大。

6. `configs/default.yaml`
   - 降低 window/batch/worker 的默认内存压力。
   - 默认让 kernel 主导 transition：低 persistence、高 kernel_mix、关闭 direct forcing。
   - 默认使用 float32 线代和 AMP。

## 建议训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m scripts.train --config configs/default.yaml
```

如果仍然被 SIGKILL，先把 `training.batch_size` 调到 1，保持 `gradient_accumulation_steps` 增大来维持 effective batch。

## 建议 ablation 顺序

1. 先跑当前 default：`forcing_scale=0`，`kernel_one_step_weight=1`，`shared_component_advection=true`。
2. 如果 kernel_one_step_loss 明显下降但 val 不下降，再逐步加入 Kalman/NLL 权重。
3. 如果 kernel_one_step_loss 也不下降，先固定 covariance/head，只检查 coordinates、state ordering、transition row/column 方向。
4. 在确认 kernel 学到传播后，再尝试 `forcing_scale=0.1` 或打开 `allow_cross_component`。
