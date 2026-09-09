# Slow Manifold

用于训练 rank-2 vanilla CTRNN，并分析 timing task 在训练过程中的 latent dynamics。目前包含两个独立任务：

- Task A：Delayed Interval Categorization
- Task B：Delayed Interval Reproduction

## 安装

要求 Windows、Python 3.13 与 [uv](https://docs.astral.sh/uv/)。在仓库根目录执行：

```cmd
uv sync --locked
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Task A：Interval Categorization

```text
S1 → [T] → S2 → [delay] → Go → short/long response
```

```cmd
:: 检查 task
uv run slow task-sanity --experiment experiments\phase0_task_sanity.yaml

:: 短程测试 / 正式训练
uv run slow train --experiment experiments\phase1_rank2_smoke.yaml
uv run slow train --experiment experiments\phase1_rank2_baseline.yaml
```

配置入口：`configs/task/interval_categorization.yaml`。

## Task B：Interval Reproduction

```text
S1 → [T] → S2 → [T_delay] → Go → [T] → response
```

三个 cue 是同一输入通道上的方波。默认 `S1=50`、`T∈[30,100)`、`T_delay∈[20,90)`；Go 后的 trial 长度为 `2T`，target 在 `Go+T` 从 0 切换为 1。

```cmd
:: 检查 train / OOD trial
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml --split ood_short
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml --split ood_long

:: CPU 短程测试 / CUDA baseline
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml
uv run slow train --experiment experiments\phase1_rank2_IR.yaml
```

loss 对 pre-Go、reproduction-wait、response 三段分别按时长归一化，再对 batch 平均。`metrics.csv` 记录 timing MAE/RMSE/bias、premature rate 和 no-response rate。配置入口为 `configs/task/interval_reproduction.yaml`，简要设计见 `notebooks/interval_reproduction_design.ipynb`。

## 常用操作

每次运行必须显式指定 experiment recipe。可临时覆盖规模或设备：

```cmd
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml --epochs 10 --batch-size 8
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml --device cuda
```

从已有 run 重算或重绘，不重新训练：

```cmd
uv run slow visualize --run RUN_DIR_HERE
uv run slow visualize --run RUN_DIR_HERE --arrow-stride 1
uv run slow visualize --run RUN_DIR_HERE --coordinate-bounds -6 6 -6 6 --grid-points 41
uv run slow visualize --run RUN_DIR_HERE --representative-epochs 0 100 1000 5000
```

默认目录由完整 task/model/train 配置指纹与 seed 决定：

```text
runs\<name>\<condition>--cfg-<fingerprint>\seed-<seed>
```

同配置、同 seed 不会覆盖已有结果。每个 run 保存 resolved config、metadata、日志、指标、checkpoints 与 diagnostics。`figures/latent_dynamics/` 保存 vector field/Jacobian 双栏 snapshots 和同步 MP4；两个旧的独立目录继续保留以兼容已有工作流。动力学图中的轨迹颜色区分任务条件，线型依次区分 interval encoding（实线）、delay（虚线）、timing/reproduction（点划线）与 response（点线）。

## 配置与测试

```text
experiments/*.yaml
  ├─ configs/task/*.yaml
  ├─ configs/model/*.yaml
  ├─ configs/train/*.yaml
  ├─ configs/analysis/*.yaml
  ├─ configs/visualization/*.yaml
  └─ configs/reporting/*.yaml
```

recipe 只组合 component config 并提供少量 override。配置优先级为：

```text
schema defaults < component configs < experiment overrides < CLI overrides
```

```cmd
uv run slow --help
uv run slow train --help
uv run slow visualize --help
uv run pytest --basetemp temp\pytest
```

研究阶段与验收标准见 `TODO.md`，开发约束见 `AGENTS.md`。
