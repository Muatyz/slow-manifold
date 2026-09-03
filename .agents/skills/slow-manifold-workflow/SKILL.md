---
name: slow-manifold-workflow
description: 管理 slow-manifold 仓库中的文献阅读、实验实现、训练、动力学诊断和结果可视化。用于本项目的研究或代码任务；不用于无关的一般性问答。
---

# Slow Manifold 工作流

本项目的研究链路为：

`文献 → 笔记 → 代码 → 数据 → 图表 → 结论 → 新问题`

开始工作前，先阅读 [AGENTS.md](../../../AGENTS.md)；涉及当前研究目标时再阅读 [TODO.md](../../../TODO.md)。只加载与当前任务相关的文献和笔记。

## 1. 按任务类型工作

- **文献与推导**：以 `references/` 原文为依据，在 `notebooks/` 记录结构化笔记、推导和待验证问题。
- **代码实现**：先搜索已有模块，沿公共接口实现；不得只在 notebook 或一次性脚本中完成。
- **训练**：使用 resolved config 和独立 RNG，先做 smoke test，再启动长任务并保存 checkpoints。
- **分析诊断**：读取已有 run/checkpoint，输出 structured results；记录定义、输入条件、阈值和数值方法。
- **可视化**：只消费 structured results，保持跨 checkpoint 的坐标、范围和色标可比较。

研究解释必须区分观测、相关性和机制证据。涉及 slow point、ghost、slow manifold 等概念时，以论文定义和 [TODO.md](../../../TODO.md) 的科学边界为准。

## 2. 环境管理

项目默认运行于 Windows，并使用 `uv`：

1. 使用根目录 `.venv`，不额外创建 conda/venv，也不直接执行 `pip install`。
2. 依赖由 `pyproject.toml` 声明、`uv.lock` 锁定；新增依赖时同步更新二者。
3. 使用 `uv run` 执行 Python，使用 `uv sync --locked` 恢复已有环境。
4. Python 版本约束在 `.python-version` 与 `pyproject.toml` 中保持一致。
5. 使用当前环境的原生 shell；README 中提供用户可直接运行的 Windows 命令。

## 3. 公共结构

```text
configs/{task,model,train,analysis,visualization}
src/slow_manifold/{tasks,models,training,analysis,visualization,utils}
scripts/{train,analyze,visualize,run_experiment}.py
experiments/
notebooks/
references/
tests/
runs/<experiment>/<seed>/
```

- `configs/` 保存可组合组件配置。
- `experiments/` 只引用组件配置并提供少量 override，不复制整块参数。
- 配置优先级为 `schema defaults < component configs < experiment overrides < CLI overrides`。
- `run_experiment.py` 只编排公共训练、分析和绘图流程，不实现平行逻辑。
- `notebooks/` 不承担核心实现；`references/` 保留原始文献。

## 4. Run 与复现

每次运行保存完整 resolved config，不只保存配置路径。run 至少包含：

```text
config.yaml
metadata.yaml
run.log
metrics.csv
status.yaml
checkpoints/
diagnostics/
figures/
```

- metadata 记录代码版本、环境、设备、dtype 和输入数据/生成规则。
- checkpoint 保存 model、optimizer、epoch/step、metrics、config 和 RNG states。
- checkpoint 至少包括 initial、periodic、best 和 final；写入应可安全恢复。
- seed 由配置控制，并为 task、model、training 和 analysis 建立可复现的独立 RNG。
- 已存在的 run 不得静默覆盖；resume 与新建 run 必须可区分。
- 相同 recipe、不同超参数的 run（如 critical learning rate 扫描）用 `tag` 区分目录：`runs/<name>/<tag>-seed-<seed>`；`tag` 由 recipe 顶层或 CLI `--tag` 提供。

`figures/latent_vector_field/` 集中保存 vector-field 产物：每个代表 epoch 的 snapshot（`epoch-*.png`）与 `latent_vector_field.mp4`。箭头采样密度由 visualization 配置 `arrow_stride` 控制（snapshot 与 movie 共用）。

decision band 染色的语义约定（与 Dinc 原文一致，勿把“模糊区”和“确定区”染反）：readout `y=tanh(z)`，**decision band 是决策模糊区** `|z|≤w`（显示阈值 `|y|≤tanh(w)`），即输出介于两类明确输出之间、尚未 commit 的区域；带内按偏向分半染色：`z∈(−w,0)`（输出为负，偏向 `−1`）红、`z∈(0,w)`（输出为正，偏向 `+1`）绿。`|z|>w` 为已明确输出 `±1` 的区域，保持速度场原状不染色。半宽 `w` 来自 visualization 配置 `decision_band_logit_half_width`（默认 1，即 Dinc 的 logit±1 约定），`w=0` 时退化为只剩决策边界。看 trajectory 落在 band 哪一侧/是否穿出带外即可读输出倾向。`representative_outputs` 面板同样叠加该 band 两半作参考。

可视化与训练解耦：`uv run slow visualize --run <run_dir>` 可从已有 run 的 checkpoints/metrics 只重算 latent dynamics 并重绘 figures，不重新训练（override `coordinate_bounds`/`grid_points`/`arrow_stride`/`representative_epochs` 时按需重算，并在 `diagnostics/visualize_rerun.yaml` 留下 override 痕迹）。无 override 时复用已存 `diagnostics/latent_dynamics.npz`。

分析坐标范围：显式 `coordinate_bounds` 优先；为 `null` 时由 trajectory 推导，并在 grid-minimum 候选贴近网格边缘（慢结构可能被截断）时按 `bounds_expansion_fraction`/`max_bounds_expansions` 自动向外扩张；`coordinate_bounds_source` 与最终范围写入 `latent_dynamics.yaml`。

动力学诊断还应记录 `F` 的时间尺度约定、`q` 的归一化、输入条件 `u`、搜索初值、阈值、去重容差及 Jacobian 方法。

### Logging

每个 run 在根目录维护统一的 `run.log`，通过项目级 logger（`src/slow_manifold/utils/logging.py`）同时输出到 terminal 与该文件；业务模块通过 `get_logger(phase)` 记录，禁止自行使用独立 `print` 作为主要状态输出。

- `run.log`：记录 run/stage 起止、周期性训练进度、checkpoint、warning、异常与 traceback 等人类可读事件；每条记录带 phase 标签。
- `metrics.csv`：update 级数值指标的唯一结构化来源，可记录 `epoch_seconds` 与累计 `elapsed_seconds`；不要在 run.log 中重复完整 metrics 行。
- `status.yaml`：记录 `running/complete/failed/interrupted`、起止时间、总耗时与失败原因摘要（`error.type`/`message`、`last_step`）；完整 traceback 仅写入 run.log。
- 默认 `INFO`，仅按 `log_every` 输出训练进度；DEBUG 可配置但默认关闭。
- ETA 使用最近若干 step 的 wall-clock 滚动中位数估算；普通计时不额外执行 CUDA synchronization。
- train/analyze/visualize 阶段均向同一 run log 追加事件，并显式标记 phase，便于回溯各阶段耗时。

## 5. 验证要求

- task：检查 timing、mask、metadata、类别平衡和 shortcut 防护。
- model：检查 rank、Euler/flow 一致性及 full/latent dynamics 对应关系。
- analysis：用有限差分或等价方法验证 Jacobian，并检查结果可重复。
- checkpoint：验证保存—加载后的输出、动力学和 RNG continuation。
- 长任务：先运行小批量、短 epoch 的端到端 smoke test；运行时提供进度和 checkpoint。

具体实现完成后，按 [AGENTS.md](../../../AGENTS.md) 的完成标准验收。
