# Agent Coding 规范

本文档约束 AI Agent 在本代码库中的长期开发行为。本项目是可维护、可复用、结果可追溯的科研代码库，不是一组一次性实验脚本。

## 0. 文档职责

- `AGENTS.md`：全仓库稳定约束。
- `.agents/skills/slow-manifold-workflow/SKILL.md`：具体科研工作流。
- `TODO.md`：当前研究目标、阶段与验收标准。
- `README.md`：面向用户的安装与使用入口。
- run 中保存的 resolved config：该次运行的实际参数记录。

若文档与用户当前明确要求冲突，先说明冲突并确认，不自行改变研究目标。

## 1. 优先复用，避免平行实现

创建文件、模块或入口前，先阅读现有结构并搜索相近功能，优先扩展已有抽象。

对于尚无实现的功能，先建立满足当前需求的最小公共路径；不要为假设中的未来需求过早设计复杂抽象。新功能具有明确复用价值时，再进入公共模块。

## 2. 避免一次性脚本

除非用户明确要求，不要为单个实验、图表或诊断创建 `figure_*.py`、`reproduce_*.py`、`*_only.py` 等专用入口。

具体研究需求优先表示为：

- 公共 runner；
- 可复用模块；
- 配置组合或少量 override；
- 轻量 experiment recipe；
- 绘图或分析规格。

## 3. 保持单一事实来源

- 配置描述“运行什么”，代码定义“如何运行”。
- schema 和默认值由核心模块维护；每次运行保存完整 resolved config。
- task、model、training、analysis 和 visualization 不重复定义同一参数或科学概念。
- simulation、flow、latent dynamics 与 Jacobian 必须派生自同一模型定义。
- experiment recipe 只组合已有配置，不复制形成第二套实验体系。

## 4. 保持职责与计算路径清晰

数据流遵循：

`config → run/checkpoint → model → analysis → structured results → visualization`

- notebook 用于学习、推导、探索和结果检查，不承载唯一的核心实现。
- analysis 负责计算并返回结构化结果，不负责 figure layout。
- visualization 消费已有结果，不重新训练或实现独立 rollout。
- 新的模拟或干预只有在科学定义确实需要时才建立。

## 5. 科研结果必须可追溯

影响结论的运行至少能够追溯到：

- resolved config 与输入数据/生成规则；
- seed 及必要的 RNG states；
- model、optimizer、checkpoint 与训练进度；
- 代码版本及工作区状态；
- Python、依赖、设备和 dtype；
- analysis 定义、阈值、输入条件与数值方法。

已有 run 不得静默覆盖。长任务应先通过小规模 smoke test，并支持进度观察、周期 checkpoint 和安全恢复。

每个 run 维护统一的 `run.log`（人类可读事件时间线，含阶段耗时与训练进度）与 `status.yaml` 生命周期（`running` → `complete`/`failed`/`interrupted`）。异常与 Ctrl+C 中断必须把错误类型、最后完成的 epoch 与简要信息写入 `status.yaml`，完整 traceback 只写入 `run.log`；不得留下停在 `running` 的半成品 run 而无法判断其停止原因。

同一 recipe 重复运行需用 `tag` 或不同 seed 区分 run 目录（`runs/<name>/<tag>-seed-<seed>`），禁止静默覆盖。可视化产物集中在 `figures/latent_vector_field/`；重绘不重训等 figure 布局与解耦规则见 SKILL §4。

## 6. 保持科学表述严谨

- 涉及论文定义或方法时，以 `references/` 中原文为主，`notebooks/` 笔记用于辅助理解。
- 区分观测、相关性和机制性证据，不根据预期选择性展示 seed。
- 不混淆 PCA projection、exact latent dynamics、slow point、ghost 与 invariant slow manifold。
- 数值结果必须携带足以解释其单位、归一化和判定条件的元数据。

## 7. 修改前与完成标准

涉及多个模块、公共接口、配置结构或新执行路径时，Agent 应先：

1. 阅读相关代码和文档；
2. 说明准备复用或扩展的抽象；
3. 给出简短方案和文件变更清单；
4. 检查是否存在更小的实现方式。

功能完成需基本满足：

- 可通过公共接口和配置使用；
- 没有不必要的重复实现或执行路径；
- 科研运行可复现且不会覆盖既有结果；
- 相关测试和 smoke test 通过；
- 必要文档已更新。

当快速完成与长期一致性冲突时，优先保持代码库与科研结果的可靠性。
