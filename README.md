# Slow Manifold

用于训练 rank-K vanilla CTRNN，并分析 timing task 在训练过程中的 latent dynamics。目前包含两个独立任务：

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

S1、S2 和 Go 与 Task B 共用同一方波 stimulus 逻辑，均写入一个标量输入通道。

```cmd
:: 检查 task
uv run slow task-sanity --experiment experiments\phase0_task_sanity.yaml

:: 短程测试 / 正式训练
uv run slow train --experiment experiments\phase1_rank2_smoke.yaml
uv run slow train --experiment experiments\phase1_rank2_baseline.yaml

:: K=3 exact-kappa / K=5 PCA-display 端到端 smoke
uv run slow train --experiment experiments\phase2_rank3_smoke.yaml
uv run slow train --experiment experiments\phase2_rank5_smoke.yaml

:: Phase 5 首轮正式训练（长任务）
uv run slow train --experiment experiments\phase5_taskA_rank3.yaml
uv run slow train --experiment experiments\phase5_taskA_rank5.yaml
```

配置入口：task timing 位于 `configs/task/interval_categorization.yaml`；loss 位于
`configs/train/rank2_baseline.yaml`。Task A 默认使用两阶段归一化 MSE：

```yaml
loss:
  name: phase_normalized_mse
  phase_weights:
    - {phase: pre_response, lambda: 1.0}
    - {phase: response, lambda: 1.0}
```

每个 trial 先分别对 Go response onset 前后按阶段长度归一化，再对 batch
平均。因此两个 `lambda` 都为 `1.0` 时，Go 前沉默与 Go 后分类各贡献一个等权 loss term。
单次实验可在 recipe 的 `overrides.train.loss.phase_weights` 中完整替换该列表；修改
lambda 会改变训练配置指纹并生成新的 run，不会覆盖原结果。

## Task B：Interval Reproduction

```text
S1 → [T] → S2 → [T_delay] → Go → [T] → response
```

三个 cue 与 Task A 一样，是同一标量输入通道上的方波。默认 `S1=50`、`T∈[30,100)`、`T_delay∈[20,90)`；Go 后的 trial 长度为 `2T`，target 在 `Go+T` 从 0 切换为 1。

```cmd
:: 检查 train / OOD trial
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml --split ood_short
uv run slow task-sanity --experiment experiments\phase0_reproduction_task_sanity.yaml --split ood_long

:: CPU 短程测试 / CUDA baseline
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml
uv run slow train --experiment experiments\phase1_rank2_IR.yaml

:: Task B 的 K=3 exact-kappa / K=5 PCA-display 端到端 smoke
uv run slow train --experiment experiments\phase2_rank3_IR_smoke.yaml
uv run slow train --experiment experiments\phase2_rank5_IR_smoke.yaml

:: Phase 5 首轮正式训练（长任务）
uv run slow train --experiment experiments\phase5_taskB_rank3.yaml
uv run slow train --experiment experiments\phase5_taskB_rank5.yaml
```

loss 对 pre-Go、reproduction-wait、response 三段分别按时长归一化，再对 batch 平均；
三个 lambda 由 Task B experiment recipe 的 `overrides.train.loss` 控制。`metrics.csv`
记录 timing MAE/RMSE/bias、premature rate 和 no-response rate。task 配置入口为
`configs/task/interval_reproduction.yaml`，简要设计见
`notebooks/interval_reproduction_design.ipynb`。

四个 Phase 5 recipe 构成首轮 paired single-seed cohort。同一任务内 K=3/K=5
共用 seed `20260903`、任务采样、训练参数、analysis 和 visualization，只有
`model.rank` 不同。Task A 使用 batch 64；Task B 沿用既有正式协议的 batch 32、
`dt=1`、`tau=10` 和三阶段等权 loss。训练仍逐 update 记录 metrics、每 10
updates validation，但周期 checkpoint 为每 1000 updates，共约 401 个分析帧；
MP4 默认关闭。单 GPU 建议逐个启动，发生中断时使用 `--resume`。

## 常用操作

每次运行必须显式指定 experiment recipe。可临时覆盖规模或设备：

```cmd
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml --epochs 10 --batch-size 8
uv run slow train --experiment experiments\phase1_rank2_IR_smoke.yaml --device cuda
```

中断或失败的 run 使用根目录中冻结的配置及最新一致 checkpoint 续训：

```cmd
uv run slow train --resume RUN_DIR_HERE
```

resume 不接受 experiment、epochs、batch size、device 或 output directory
覆盖。它恢复 model、optimizer、task/rollout RNG 与累计 metrics；如果中断发生在
metric 已写入但 checkpoint 尚未保存之间，会原子截断 checkpoint 之后的 metric
尾部再继续。已完成的 run 仍拒绝 resume，新训练仍拒绝覆盖已有目录。

validation 频率由 train config 控制，默认每 10 个 update 执行一次：

```yaml
# configs/train/rank2_baseline.yaml
validation_every: 10
```

experiment recipe 可针对单次实验覆盖：

```yaml
overrides:
  train:
    validation_every: 20
```

epoch 0、final 和所有保存 checkpoint 的 epoch 始终执行 validation。非 validation epoch 仍在 `metrics.csv` 中记录 train loss、gradient norm 与耗时，validation 字段留空。

从已有 run 重算或重绘，不重新训练：

```cmd
:: 默认读取该 run 所记录 experiment 的当前 analysis/visualization 配置
uv run slow visualize --run RUN_DIR_HERE

:: 改用训练启动时配置 / 最近一次成功重绘配置
uv run slow visualize --run RUN_DIR_HERE --config-source original
uv run slow visualize --run RUN_DIR_HERE --config-source last

:: 显式指定新的 recipe；只读取其中的 analysis/visualization
uv run slow visualize --run RUN_DIR_HERE --experiment experiments\phase1_rank2_baseline.yaml

:: 临时 CLI override（优先级最高）
uv run slow visualize --run RUN_DIR_HERE --arrow-stride 1
uv run slow visualize --run RUN_DIR_HERE --arrow-length-fraction 0.0075 --arrow-width 0.0012
uv run slow visualize --run RUN_DIR_HERE --snapshot-dpi 300 --movie-dpi 120
uv run slow visualize --run RUN_DIR_HERE --render-movies
uv run slow visualize --run RUN_DIR_HERE --trajectory-line-width 0.8
uv run slow visualize --run RUN_DIR_HERE --coordinate-bounds -6 6 -6 6 --grid-points 41

:: 人工替换自动选择的 representative checkpoints
uv run slow visualize --run RUN_DIR_HERE --representative-epochs 0 100 1000 5000
```

`current`（默认）固定使用原 run 的 task、model、checkpoints 与 metrics，只从当前 recipe 更新 analysis/visualization；其路径不可用时用 `--experiment` 指定。`original` 严格复现训练启动时设置，`last` 复用最近一次成功阶段设置。CLI override 会写入完整阶段配置，但下次默认 `current` 不会继承；需要继承时使用 `--config-source last`。

默认在训练结束后根据 validation loss 自动选择最多五个阶段。平滑曲线只定位阶段：中间阶段显示局部最接近平滑趋势的实际 checkpoint，`mature` 显示最佳平滑区域内 raw validation loss 最好的 checkpoint，避免把窗口中心的瞬时坏点画入 representative outputs。阶段锚点、候选窗口和最终 epoch 均写入 `diagnostics/checkpoint_selection.yaml`。`abrupt`、`quasi_plateau` 和 `mature` 都是描述性标签，不代表 optimum、任务已经学会或某种动力学机制。人工指定的 epoch 必须存在对应 checkpoint，且不会再被强制追加 final epoch。

```yaml
# configs/analysis/training_dynamics.yaml
representative_selection:
  mode: auto                  # 改为 manual 时填写 manual_epochs
  metric: validation_loss
  max_epochs: 5
  smoothing_points: 5         # 平滑及局部选点窗口；单位为 checkpoint 点数
  transition_radius_points: 3
  abrupt_effect_mad: 5.0
  abrupt_slope_ratio: 5.0
  plateau_points: 50
  plateau_max_relative_improvement: 0.02
  manual_epochs: []
```

K=2 时，改变 selection 规则或 `--representative-epochs` 只重选帧和重绘。K≥3 的 full-state slow-point refinement 只在代表性 checkpoints 上执行，因此改变所选 epochs 会触发相应 structured diagnostics 重算，并写入 analysis 指纹。

默认目录由完整 task/model/train 配置指纹与 seed 决定：

```text
runs\<name>\<condition>--cfg-<fingerprint>\seed-<seed>
```

同配置、同 seed 不会覆盖已有结果。根 `config.yaml` 是不可修改的训练记录；`diagnostics/config.yaml` 与 `figures/config.yaml` 分别保存最近一次成功使用的完整 analysis/visualization 配置、来源与指纹。analysis 指纹不变时直接复用结构化 diagnostics，变化时从 checkpoints 重算；所有阶段事件追加到 `run.log`。

`figures/latent_dynamics/` 是统一入口。K=2 时保存 vector field/Jacobian 双栏 snapshots，两个独立目录继续保留单栏图；其轨迹颜色区分任务条件、线型区分任务阶段。K≥3 时不构造难以解释的 3-D quiver，而保存 trajectories、optimized slow points（按 `q_x` 着色）、同一批点的 full-state spectral abscissa 三栏 3-D snapshots；`q_x` 使用只由当前展示点决定的鲁棒对数色标，谱色标以 0 为中心，三栏共享逐 checkpoint 推导的坐标范围，避免被其他训练阶段的极值压缩。最后一个代表性 checkpoint 还会在 `latent_dynamics/single_trajectory/` 输出一张中位 interval 附近的单 trajectory 示意图，并按该轨迹单独取 bounds。轨迹段以颜色（并冗余使用线型）区分 interval encoding、delay、timing/reproduction 与 response，终点 marker 再编码 short/long 或 interval 条件。实际 `u≠0` 的 cue-driven 位移段在 Task A 中显示为黄色加黑色描边，在 Task B 中按 Ramesan 风格显示为无黑色包络的红线；3-D 图的红色 cue overlay 使用独立 alpha 和线宽缩放，减少遮挡。MP4 仅在 `render_movies: true` 时生成。

坐标策略没有 K=4 缺口：K=3 直接显示 exact `κ=Nᵀx` 的三个坐标；K>3（包括 K=4）用 task-trajectory `κ` states 拟合 PC1–3。主 slow-point 搜索按任务阶段分层抽取 trajectory full states，连续最小化 `q_x=½‖F_x‖²`，以 `q_x≤10⁻⁴` 接受，并在相同优化端点计算解析 full-state Jacobian 与 `max Re(λ(J_x))`；PCA 只用于最终显示。原有 K 维 κ 邻域的 `qκ`/Jacobian samples 仍保存在 structured diagnostics 中作为探索性候选，不能单独作为 fixed point、ghost 或 invariant slow manifold 的证据。

κ 平面的网格与形状由 analysis config 控制：

```yaml
# configs/analysis/training_dynamics.yaml
grid_points: 61                 # 每轴采样数；色块总数约为 61²
coordinate_bounds: null         # 或显式写 [x_min, x_max, y_min, y_max]
square_coordinate_bounds: true  # 扩展较短轴，保持正方形且不拉伸
evaluation_trials:              # Task A: 阈值 5 两侧各取两个 T
  - {interval: 3.0, delay: 2.0, s1_onset: 1.0}
  - {interval: 4.5, delay: 2.0, s1_onset: 1.0}
  - {interval: 5.5, delay: 2.0, s1_onset: 1.0}
  - {interval: 7.0, delay: 2.0, s1_onset: 1.0}
```

Task B 在 experiment override 中使用 `T=30,50,75,99`。修改 `evaluation_trials` 会改变 structured diagnostics 并触发重算。`grid_points` 同时作用于 speed 和 Jacobian 色块；提高到 `81` 会更细，但网格计算量约按 `grid_points²` 增长。

轨迹显示由 visualization config 控制；主线、基线、cue overlay、Task A cue 描边和终点 marker 会同比缩放：

```yaml
# configs/visualization/training_dynamics.yaml
loss_y_scale: log              # loss 纵轴；可改为 linear
trajectory_line_width: 0.9
trajectory_3d_alpha: 0.62
trajectory_3d_cue_alpha_scale: 0.55
trajectory_3d_cue_line_width_scale: 0.8
trajectory_3d_view_elevation: 26.0
trajectory_3d_view_azimuth: -68.0
single_trajectory_enabled: true
single_trajectory_interval_quantile: 0.5
snapshot_dpi: 240              # PNG；用于论文/放大检查
movie_dpi: 120                 # MP4；与 PNG 解耦以控制编码成本
render_movies: false           # 高开销实验默认只保存静态图
```

静态图和视频分开配置分辨率；需要 movie 时在 experiment visualization override
中显式启用，或重绘时传入 `--render-movies`。旧 run/recipe 若只包含
`vector_field_dpi`，重绘时仍会把它作为兼容 fallback。高维 smoke 使用
`15×5 in @ 200 DPI`，输出约 3000×1000 像素，不再为了缩短 MP4 编码时间把
PNG 一并降到 50 DPI。

已有 run 可不训练直接重算或重绘：

```cmd
uv run slow visualize --run RUN_DIR_HERE --grid-points 81 --square-coordinate-bounds
```

训练 rollout 的随机性由 train config 控制：

```yaml
# configs/train/rank2_baseline.yaml
rollout:
  initial_state: {distribution: tanh_normal, mean: 0.0, std: 0.1}
  neural_noise:
    distribution: normal
    mean: 0.0
    std: 1.0e-3
    injection: activation_input
```

即 `x(0)=tanh(z), z~N(0,0.1²)`，并在每个 Euler step 将 `ε~N(0,10⁻⁶)` 加入 `tanh` 的输入。缺少 `rollout` 区块的历史 run 解析为两个 `std=0`，保持原来的零初态、无噪声行为。validation 默认复用一次采样的初态且关闭 neural noise；vector field/Jacobian 始终描述无采样噪声的确定性 drift。

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
