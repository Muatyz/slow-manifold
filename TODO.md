# Rank-2 RNN Timing Dynamics：实施前研究计划

> 状态（2026-09-07）：Task A 已完成一个 seed 的正式 baseline；Task B（Interval Reproduction）已完成 generator、phase-normalized loss、行为指标与端到端 smoke pipeline，正式训练尚未开始。multi-seed、trajectory/global 初始化、eigenvectors 与 full-state ghost diagnostics 尚未完成。

## 1. 研究目标

训练一个 recurrent connectivity 受 rank-2 约束的 vanilla continuous-time RNN，完成 Delayed Interval Categorization，并研究：

1. 网络如何编码并保持两个脉冲之间的时间间隔；
2. 训练过程中二维 vector field 如何重组；
3. fixed points、slow points 或 ghost candidates 是否组织 task-relevant trajectories；
4. slow structure 的局部 Jacobian spectrum 是否与行为学习及 temporal generalization 有关。

研究路线：

`behavior → exact 2D latent dynamics → fixed/slow points → spectrum → learning dynamics → generalization`

本项目不是复现某篇论文，也不预设训练结果必须符合 ghost mechanism、slow manifold或特定分岔。

## 2. Baseline 选择依据

- **任务不与 Dinc 直接重合**：Delayed Interval Categorization 需要 timing，但不同于 delayed activation、DCD 和 DMTS。
- **与 Ramesan 保持问题层面的联系**：网络仍需完成 interval encoding、delay memory 和 cue-triggered readout，只是行为目标从 interval reproduction 改为 threshold judgment。
- **控制入门复杂度**：二分类比时间复现更容易先建立可靠的训练—诊断闭环。
- **rank-2 服务于可解释性**：它保留 vanilla RNN 单元动力学，同时提供精确闭合的二维 latent system，便于直接观察训练期间的 vector-field reorganization。
- **full-rank 是后续对照**：rank-2 pipeline 稳定后，在相同任务上训练 full-rank vanilla RNN，检验观察到的机制是否依赖低秩约束。

## 3. Baseline 规范

### 3.1 Delayed Interval Categorization

Trial：

`pre-S1 jitter → S1 → [T] → S2 → [T_delay] → Go → response`

- `T=t(S2)-t(S1)`。
- `T<T_c` 为 short，target 为 `-1`；`T>T_c` 为 long，target 为 `+1`。
- `T=T_c` 的标签或采样排除规则必须在配置中明确。
- Go 之前 target 为 `0`；仅在 Go 后的 response window 报告类别。
- 随机化 S1 onset 和 `T_delay`，避免网络依赖绝对 trial time。
- short/long 两类平衡采样，并显式保存 `S1/S2/Go time`、`T`、`T_delay` 和 class。
- 初始建议值为 `T∼U(2,8)`、`T_c=5`、`T_delay∼U(1,4)`；时间单位、pulse width 和 response window 在 Phase 0 冻结，代码中不得写死。
- 使用显式 `loss_mask` 的 MSE；response-only loss 与是否惩罚 pre-Go output 由配置控制。

主要行为指标：

- validation loss 和 classification accuracy；
- `P(long|T)` psychometric curve；
- estimated decision boundary 及其相对 `T_c` 的偏差；
- 对 S1 jitter 和 `T_delay` 的条件化准确率。

### 3.2 Rank-2 vanilla CTRNN

完整状态动力学：

`τ dx/dt = -x + tanh(Wx + W_in u + b)`，其中 `x∈R^N`。

Recurrent matrix：

`W=MNᵀ`，`M,N∈R^(N×2)`，因此 `rank(W)≤2`。

定义 `κ=Nᵀx`，可得到精确闭合的二维动力学：

`τ dκ/dt = -κ + Nᵀ tanh(Mκ + W_in u + b)`。

要求：

- baseline 使用 `N=64`、`tanh` 和 Euler integration；数值均配置化。
- 训练 simulation、完整状态 `F_x`、latent `F_κ` 和 Jacobian 必须来自同一动力学定义。
- 二维 `κ` 是模型的精确 latent coordinate，不是对 neural activity 做 PCA。
- 初始化和尺度必须避免 `M/N` factorization 带来的数值失衡；监测两者范数及 `W` 的 singular values。

### 3.3 Interval Reproduction（Task B）

- 沿用同一个 rank-2/tanh CTRNN；S1、S2、Go 共用一个方波输入通道。
- Go 后观察 `2T`：前 `T` 为 reproduction-wait，后 `T` target 为 1。
- pre-Go、wait、response 的 MSE 分别按阶段时长归一化，再按配置权重相加并对 batch 平均。
- 用首次持续阈值穿越报告 timing MAE/RMSE/bias、premature rate 与 no-response rate。
- [x] task、loss、metrics、training/analysis/visualization 与重绘 smoke pipeline。
- [ ] 正式训练与行为验收；minimum rank 和 tolerance accuracy 待观察结果后定义。

## 4. 分阶段工作与验收

下列 `[x]` 表示对应实现已完成并通过自动测试或 smoke test；依赖正式实验结果的验收项，只有在实际运行并检查结果后才标记完成。

### Phase 0：冻结规范与 task sanity

- [x] 冻结时间单位、`dt/τ`、方波宽度、response window 和 loss mask。
- [x] 明确 `T=T_c` 的处理，并划分 train/validation/OOD conditions。
- [x] 随机生成并绘制 trials，检查输入、target、mask、metadata 和类别平衡。
- [x] 检查改变 S1 onset 或 `T_delay` 不会改变同一个 `T` 的标签。

验收：task generator 的全部时间关系和边界条件通过自动测试；尚不训练网络。

### Phase 1：训练 rank-2 baseline

- [x] 通过公共 runner 跑通短程 CPU/CUDA smoke training，并生成完整 run 目录。
- [x] 用一个 seed 完成正式 baseline 训练；单 seed 仅作工程验收。
- [ ] 建立小型 multi-seed cohort。
- [x] 记录 train/validation loss、accuracy、gradient norm、learning rate、`‖M‖/‖N‖/‖W‖` 和两个 singular values。
- [x] 保存 initial、periodic、best、final checkpoints，以及 model/optimizer/config/seed/metrics 和 RNG state。
- [ ] checkpoint 频率足以解析潜在的快速行为跃迁。
- [ ] 用 psychometric curve 检查网络确实利用 interval，而不是记忆少数离散模板。

验收：checkpoint 可恢复相同动力学；至少一个 seed 在随机 S1 onset 和 `T_delay` 的 held-out trials 上稳定分类。

### Phase 2：分析 trained-network dynamics

对每个固定输入条件定义：

`q_κ(κ)=1/2‖F_κ(κ,u)‖²`。

优先分析 autonomous `u=0`，再按需要比较 S1、S2 和 Go 条件，因为输入会重构 vector field。

- [x] 在跨 checkpoint 对齐的二维网格评估 `F_κ` 与 normalized speed，并保存结构化 NPZ/YAML。
- [x] 在同一网格解析计算 exact latent Jacobian，保存完整复特征值与 `max Re(λ)` spectral-abscissa map。
- [x] 绘制初步 quiver、speed background 和 short/long task trajectories。
- [x] 对 refined minima 显式计算并保存 `q_κ=1/2‖F_κ‖²`，统一分类阈值定义。
- [ ] 用 task-trajectory states 与 global/random points 分别初始化 `q_κ` minimization。
- [x] 从八邻域网格候选初始化、连续优化、去重并分类 latent fixed/slow points 与 latent ghost candidates。
- [ ] 保存位置、`q`、speed、初始化来源、优化状态、Jacobian、eigenvalues 和 eigenvectors。
- [ ] 对 latent candidate 附近的 task-relevant full state 额外最小化 `q_x(x)=1/2‖F_x(x,u)‖²`。
- [ ] 只有 full-state slow point 及完整 Jacobian 都支持 transverse stability 时，才标记为 ghost candidate。
- [ ] `q` 阈值配置化并进行敏感性检查，不直接照搬 Ramesan 的 `10^-4`。

验收：一张 phase portrait 可以同时展示 vector field、task trajectories、fixed points、slow points 和局部稳定性；每个标签均可追溯到结构化诊断数据。

### Phase 3：分析 learning dynamics

rank-2 factorization 存在 `M→MA, N→NA^(-T)` 的 gauge freedom。跨 checkpoint 比较时，不能直接把原始 `κ` 坐标逐帧拼接。

- [x] 从每个 checkpoint 的完整 `W` 构造 rank-2 SVD diagnostic basis。
- [x] 对初始 basis 固定 sign，并对相邻 checkpoint 做 orthogonal Procrustes alignment。
- [ ] 增加近简并 singular values 的检测、告警与稳健性测试。
- [x] 跨帧固定坐标范围、evaluation trials 和 speed color scale。
- [x] 支持在 initial、配置指定的 intermediate 和 final checkpoints 生成 output panels 与初步 MP4。
- [x] 在同一 snapshot/MP4 中同步展示 vector field 与 Jacobian spectral abscissa。
- [x] 轨迹使用颜色编码任务条件、线型编码任务阶段，并在独立图与联合图中保持一致。
- [x] 联合绘制 loss、validation loss 与 gradient norm，并标注代表性 epochs。
- [ ] 在正式训练后定位 behavioral transition，并重复完整 Phase 2 diagnostics。
- [ ] 联合比较 accuracy、`min(q)`、slow-region extent、local spectrum 和 residence time。
- [ ] 若出现 abrupt learning，先定义可复现的行为判据，再检查 slow structure 与行为改变的时间顺序。

验收：描述性 MP4 管线已打通；仍需先完成正式训练和静态 slow-point/Jacobian diagnostics，再将 movie 用作科学结果。

### Phase 4：temporal generalization

categorization 的 generalization 与 interval reproduction 不完全等价，因此只作探索性扩展：

- [ ] 在训练区间内使用更密集、未见过的连续 `T` 测试 psychometric curve。
- [ ] 测试训练范围外的 `T`，但注意远离阈值的样本可能更容易，不能只报告总体 OOD accuracy。
- [ ] 独立扩展 S1 onset 和 `T_delay` 的测试范围，检验相对时间编码和 delay robustness。
- [ ] 跨 seed 比较 decision-boundary stability 与 slow-region extent、speed、spectrum 和 residence time。
- [ ] 不把相关性自动解释为 Ramesan 结论在新任务上的复现或因果证明。

验收：分别报告 interval、delay 和 onset 三类泛化；明确描述性、相关性与机制性证据的边界。

### Phase 5：full-rank extension

仅在 rank-2 milestone 完成后进入：

- [ ] 保持 task、训练 protocol 和主要超参数尽可能一致，只移除 rank constraint。
- [ ] slow-point search 和 Jacobian analysis 在完整状态空间进行；PCA 只用于展示。
- [ ] 比较行为表现、训练稳定性、slow structures 与 seed-to-seed variability。
- [ ] 判断 rank-2 中观察到的机制是普遍现象还是低秩约束的结果。

## 5. 文献诊断如何使用

- **Dinc**：采用 `q` minima、完整 Jacobian spectrum、loss/gradient 联合轨迹；不因出现 loss drop 就宣称 ghost mechanism。
- **Ramesan**：借鉴 task-based slow-point initialization、input-conditioned vector fields、slow-set organization 和 generalization 分析；不声称当前分类任务复现其 interval-reproduction 结论。
- **Ságodi**：用于约束 slow manifold/continuous attractor 的表述；需要切向慢漂移、法向一致收缩和局部不变性证据。

## 6. 最小工程约束

- 核心逻辑按 `task / model / training / analysis / visualization` 分离。
- notebook 用于文献学习、task sanity、探索和结果检查；核心算法不只存在 notebook cell 中。
- 分析函数返回结构化结果，plotter 不负责 slow-point search 或 Jacobian 计算。
- 所有实验由 config 和 seed 可复现；结果保留 config、metrics、checkpoints、diagnostics 和 figures。
- 现有三个文献 notebook 保留原样。
- 在 Phase 0 完成前，不预先锁定庞大目录树或逐函数 API。

## 7. 科学表述边界

- low speed along a trajectory ≠ slow point。
- latent slow point ≠ full-state ghost。
- slow points 的集合 ≠ invariant slow manifold。
- near-zero eigenvalue ≠ criticality 或 saddle-node bifurcation 的充分证据。
- slow point ≠ ghost；ghost 还需要 transverse stability。
- rank-2 exact latent dynamics ≠ full-rank 网络也具有同样的二维闭合系统。
- 不挑选性地只展示符合预期的 seed。

## 8. 暂缓内容

第一个 milestone 不包括：

- full-rank、GRU、LSTM 或其他 timing tasks；
- 完整 bifurcation continuation、XPPAUT/BifurcationKit 集成；
- Dinc 的 confidence intervention/no-learning-zone rescue；
- 大规模超参数扫描、manifold surface fitting 和生物学解释；
- 在静态 checkpoint diagnostics 验证前制作精制 movie。

## 9. 第一个 milestone

1. 一个经过 sanity check 的 Delayed Interval Categorization generator；
2. 一个可复现训练成功的 rank-2 vanilla CTRNN；
3. trained network 的精确二维 phase portrait 与 full-state ghost validation；
4. initial/intermediate/final checkpoints 的坐标一致诊断；
5. 一份简短报告，说明行为、动力学证据、尚不能支持的解释和下一步选择。

完成后，再决定优先制作完整 training movie、开展 multi-seed generalization，还是进入 full-rank extension。
