# Slow Manifold

用于研究 rank-2 vanilla CTRNN 如何完成 Delayed Interval Categorization，以及训练期间 latent vector field 如何演化。

当前已完成 task、训练、checkpoint 和 Dinc-style figures 管线。只运行过 smoke test，尚无可用于科学结论的正式训练结果。

## 快速开始

要求：Windows、Python 3.13、[uv](https://docs.astral.sh/uv/)。

在仓库根目录安装锁定依赖：

```cmd
uv sync --locked
```

项目使用根目录 `.venv`；不要直接运行 `pip install` 修改环境。

先检查任务生成是否正确：

```cmd
uv run slow task-sanity --experiment experiments\phase0_task_sanity.yaml
```

再执行 2-epoch smoke test：

```cmd
uv run slow train --experiment experiments\phase1_rank2_smoke.yaml
```

确认 figures、metrics 和 checkpoints 均已生成后，再启动正式训练：

```cmd
uv run slow train --experiment experiments\phase1_rank2_baseline.yaml
```

正式 baseline 使用 CUDA、20000 次 online gradient updates（当前配置名为 `epochs`）和 batch size 64；PyTorch 由 uv 锁定到官方 CUDA 13.0 wheel。安装后可检查 GPU：

```cmd
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

需要临时回退到 CPU 时：

```cmd
uv run slow train --experiment experiments\phase1_rank2_smoke.yaml --device cpu --output-dir runs\phase1_rank2_smoke_cpu\seed-20260903
```

默认输出目录由 recipe 中的 `name` 和 `seed` 自动生成为 `runs\<name>\seed-<seed>`。程序不会覆盖已有 run；`--output-dir` 只用于临时覆盖保存位置。

同一 recipe 需要跑多次且只改超参数时（例如 Dinc 讨论过的 critical learning rate 扫描），用 `--tag` 分离目录；recipe 顶层也可写 `tag:`：

```cmd
uv run slow train --experiment experiments\phase1_rank2_baseline.yaml --tag lr1e-4
# -> runs\phase1_rank2_baseline\lr1e-4-seed-20260903
```

训练完成后可**不重新训练**只重绘 figures（改箭头密度/坐标范围/网格分辨率等）：

```cmd
uv run slow visualize --run runs\phase1_rank2_baseline\seed-20260903 --arrow-stride 1
uv run slow visualize --run runs\phase1_rank2_baseline\seed-20260903 --coordinate-bounds -6 6 -6 6 --grid-points 33
```

查看完整参数：

```cmd
uv run slow --help
uv run slow train --help
uv run slow visualize --help
```

## 任务定义

```text
pre-S1 jitter -> S1 -> [T] -> S2 -> [T_delay] -> Go -> response
```

- `T < T_c`：short，target `-1`
- `T > T_c`：long，target `+1`
- `T = T_c`：不采样
- S1、S2 和 Go 为可配置宽度、幅值的方波
- S1 onset 与 `T_delay` 随机化，short/long 平衡采样
- Go 前 target 为 `0`，Go 方波结束后进入 response window

输入为 `inputs[B,T,3]`，三个通道依次是 S1、S2、Go；`target` 和 `loss_mask` 为 `[B,T,1]`。

## 配置入口

| 文件 | 主要参数 |
| --- | --- |
| `configs/task/interval_categorization.yaml` | `threshold`、方波、loss mask、train/validation/OOD 区间 |
| `configs/model/rank2_ctrnn.yaml` | `N`、rank、`dt/τ`、dtype、初始化 |
| `configs/train/rank2_baseline.yaml` | epochs、batch size、optimizer、device、checkpoint 间隔 |
| `configs/analysis/training_dynamics.yaml` | 代表 epoch、vector-field 网格与坐标范围、bounds 自动扩张、输入条件、evaluation trials |
| `configs/visualization/training_dynamics.yaml` | figure 尺寸、MP4 fps、codec、箭头密度(`arrow_stride`)、decision band 宽度 |

`experiments/phase1_rank2_baseline.yaml` 组合这些配置并提供少量 overrides。配置优先级为：

```text
component configs < experiment overrides < CLI overrides
```

每次运行都必须显式传入 `--experiment`；无需手动指定输出目录。不同用途应使用不同的 experiment recipe，例如正式训练使用 `phase1_rank2_baseline.yaml`，smoke test 使用 `phase1_rank2_smoke.yaml`。

默认阈值和方波示例：

```yaml
threshold: 5.0
stimulus:
  waveform: square
  width: 0.2
  amplitude: 1.0
```

`width` 和 `threshold` 必须与 `dt` 对齐。修改阈值时，也要保证每个 split 的 `short_interval` 和 `long_interval` 分别位于阈值两侧。

Task sanity 支持：`train`、`validation`、`ood_interval`、`ood_delay` 和 `ood_onset`，例如：

```cmd
uv run slow task-sanity --experiment experiments\phase0_task_sanity.yaml --split ood_interval --output-dir runs\phase0_task_sanity\ood-interval
```

## Run 产物

```text
config.yaml                         实际使用的 resolved config
metadata.yaml                       seed、环境、Git、device、dtype 与 RNG
run.log                             人类可读事件时间线:阶段起止、训练进度与 ETA、异常
metrics.csv                         loss、accuracy、gradient、低秩参数与逐步 wall-clock
status.yaml                         run 生命周期与结束状态
checkpoints/                        initial、periodic、best、final
diagnostics/latent_dynamics.npz     结构化 vector field、trajectory、input/output 与 valid mask
diagnostics/latent_dynamics.yaml    flow、速度、输入和坐标定义
figures/loss_and_gradient.png       loss/gradient norm—epoch
figures/representative_outputs.png  代表 epoch 的 network/target 对比
figures/latent_vector_field/        代表 epoch 的 vector-field snapshot + MP4
```

`metrics.csv` 中 recurrent gradient norm 定义为 `||∇M L||F + ||∇N L||F`；同时记录全参数 gradient norm、`||M||/||N||/||W||` 和两个 singular values。每行还包含 `epoch_seconds`（本次 forward/backward/validation 的 wall-clock）与累计 `elapsed_seconds`，便于吞吐估算与时间外推。

`run.log` 由项目级 logger 统一写入，同时镜像到终端，记录带 phase 标签（`run`/`train`/`analysis`/`visualization`）的事件，默认仅按 `log_every` 输出训练进度并给出基于滚动中位数的 ETA；逐步数值不重复进入日志，唯一来源是 `metrics.csv`。

`status.yaml` 维护 run 生命周期（`running` → `complete`/`failed`/`interrupted`）：

```yaml
status: running
started_at_utc: ...
# 结束后追加:
finished_at_utc: ...
elapsed_seconds: 372.8
# failed/interrupted 时:
last_step: 3847            # 从 metrics.csv 恢复的最后一个完成的 epoch
error:
  type: RuntimeError       # 例如 ValueError(配置错误)、RuntimeError(硬件)、KeyboardInterrupt
  message: CUDA out of memory
```

完整 traceback 只写入 `run.log`，不塞进 YAML。

Movie 与 vector-field snapshots（每个代表 epoch 一张 PNG）默认展示 `u=[0,0,0]` 的 autonomous latent flow，并使用跨 checkpoint 对齐的 SVD/Procrustes 坐标。读入采用 `y=tanh(z)`（`z` 为 readout logit），**decision band 指决策模糊区**：`|z|≤w`（显示为 `|y|≤tanh w`）之间的过渡带，输出尚未确定地 commit 到哪一类。带内按偏向染色：`z∈(−w,0)`（输出为负、偏向 class `−1`/short/`T<T_c`）染红，`z∈(0,w)`（输出为正、偏向 class `+1`/long/`T>T_c`）染绿；而 `|z|>w` 表示已明确输出 `−1` 或 `+1` 的区域，保持速度场原状不染色。半宽 `w` 由 `decision_band_logit_half_width`（默认 1，即 Dinc 的 logit±1 约定；对 sigmoid 即为 `σ(−1)..σ(1)`）控制，`w=0` 时只剩决策边界。因此观察一条 trajectory 落在 band 的哪一侧/是否穿出带外，即可读出其输出朝 `−1` 还是 `+1`。两类 trajectory 与其目标类别同色（`−1` 红、`+1` 绿）。`representative_outputs.png` 每个 trial 顶部先以黑色方波（按 S1/S2/Go 三条水平带错开）给出输入脉冲示意（Ramesan Fig. 1a 风格），下方各 epoch 面板为 output 与 target（target 实线、output 虚线，`−1` 红、`+1` 绿），并叠加同语义的 decision band 半带（`y∈(−tanh w,0)` 红、`y∈(0,tanh w)` 绿）。箭头密度由 `arrow_stride` 控制（每个 flow-grid 点隔 n 画一个箭头）；坐标范围由 `coordinate_bounds` 显式指定，或由轨迹推导，并在 grid-minimum 候选贴近边缘时按 `bounds_expansion_fraction`/`max_bounds_expansions` 自动向外扩张。显式 `coordinate_bounds` 优先于自动扩张。紫色叉号只是离散网格上的 speed-minimum candidates；不能仅凭它们、低速区域或 loss jump 宣称形成了 ghost。

## 测试

```cmd
uv run pytest
```

研究计划、阶段验收与科学表述边界见 `TODO.md`；开发约束见 `AGENTS.md`。
