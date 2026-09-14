# Adam + BPTT 为什么保持低秩连接：meeting 第一个问题的证明

日期：2026-09-09. 对应 `meeting_notes.md` 第 1 个问题. 本文是基于原文和当前实现的独立推导, 不是两篇论文已有定理的转述. 

**结论：Adam 本身不保持矩阵的秩. 当前模型保持 $\operatorname{rank}(W) \le K$, 是因为可训练参数始终是两个固定形状的因子, 且每一步都定义 $W = AB^\top$. BPTT 决定因子的梯度, Adam 决定因子的更新；秩上界由参数化在任意有限步长下严格保证. **

## 1. 原文依据与适用边界

### Mastrogiuseppe & Ostojic (2018)

原文：[Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks](https://doi.org/10.1016/j.neuron.2018.07.003). 仓库 PDF 位于 `references/Mastrogiuseppe和Ostojic - 2018 - Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks.pdf`. 

主文 p.610, 式 (1)–(2) 使用

$$
\dot h=-h+J\phi(h)+I,\qquad J=g\chi+P,\qquad P=\frac{mn^\top}{N}.
$$

因此 rank-1 约束首先指结构化部分 $P$. 推广至 K 个外积, $P = \sum_k m_k n_k^\top/N$ 的秩至多为 K；当 $g \neq 0$ 时, 不能据此断言整个 $J$ 的秩至多为 K. 该文提供低秩连接的结构定义及其动力学分析, 不是“Adam 自动保秩”的证明. 

### Ger & Barak (2026)

原文：[Learning Reveals Invisible Structure in Low-Rank RNNs, arXiv:2605.04115v1](https://arxiv.org/html/2605.04115v1). 仓库 PDF 位于 `references/Ger和Barak - 2026 - Learning reveals invisible structure in low-rank RNNs.pdf`. 

§3 式 (4) 将有效 recurrent coupling 写成 $uv^\top/N$, 并明确以 $\{m,u,v,z\}$ 四个向量为可训练参数. §2 式 (3)、§3 式 (8)–(10) 与 Appendix A.2 分析这些参数在梯度流下诱导的 overlap 动力学：

$$
\nabla_\theta L=D(\theta)^\top\nabla_\sigma L,
\qquad
\dot\sigma=-D(\theta)D(\theta)^\top\nabla_\sigma L.
$$

其中 $D = \partial\sigma/\partial\theta$. 这也说明“在因子空间优化”不能直接替换为“在约化变量空间做普通梯度下降”. 

Appendix A.5 的梯度流守恒量和 Fig.7 中 Adam 对其的破坏, 是另一种性质；不能把守恒量被破坏理解为 $uv^\top/N$ 不再低秩. Appendix B.4.1 / Fig.8 还显示, Adam 可使其非线性 overlap 理论所需的高斯近似失效. 本文的代数秩证明不需要这些梯度流、大 N 或高斯假设. 

### 与当前代码的区别

当前仓库使用

$$
\tau\dot x=-x+\tanh(Wx+Uu+b),
$$

两篇论文主要使用 $-h + W\phi(h) + \mathrm{input}$ 形式. 非线性的位置不同, 不能直接照搬其 state Jacobian 或 BPTT 公式. 下面从当前代码的 Euler 更新重新推导；因子乘积的秩界对两种形式都成立. 

## 2. 记号及当前模型

为避免神经元数 N 与代码中的右因子 `n` 混淆, 记：

- $N$：神经元数；$K \le N$：秩上限. 
- $A, B \in \mathbb{R}^{N\times K}$：分别对应 `model.m` 与 `model.n`. 
- $W = AB^\top \in \mathbb{R}^{N\times N}$；当前实现没有额外 $1/N$ 系数. 
- $U \in \mathbb{R}^{N\times d}$：输入权重；$b \in \mathbb{R}^N$：偏置. 
- $c \in \mathbb{R}^K$、$d \in \mathbb{R}$：单输出时的 latent readout 权重与偏置. 
- $t$：trial 内离散时间；$s$：Adam 优化步数；两者不能混用. 

固定一次优化步中的参数, 令 $\alpha = \Delta t/\tau$. 对 $t = 0, \ldots, T-1$：

$$
a_t=AB^\top x_t+Uu_t+b,
\qquad
x_{t+1}=(1-\alpha)x_t+\alpha\tanh(a_t).
\tag{1}
$$

代码在每次状态更新之后读出, 故输出索引为 $t = 1, \ldots, T$：

$$
\kappa_t=B^\top x_t,\quad
z_t=c^\top B^\top x_t+d,\quad
\hat y_t=\tanh(z_t),\quad
L=\sum_{t=1}^T w_t(\hat y_t-y_t^*)^2.
\tag{2}
$$

$w_t$ 是固定的数据权重, 包含 mask 和归一化分母；多 trial 情况再对 batch 求和. 这覆盖当前 weighted_mean 和 batch_mean 两种 loss reduction. 假设初始状态与可训练参数独立, 输入、target 和 mask 在反传中固定. 以下先推导数据损失, weight decay 在 §5 单独处理. 

## 3. 从 BPTT 求 recurrent 路径的梯度

定义

$$
D_t=\operatorname{diag}(1-\tanh^2(a_t)),
\qquad
J_t=\frac{\partial x_{t+1}}{\partial x_t}
=(1-\alpha)I_N+\alpha D_tW.
\tag{3}
$$

注意 $D_tW$ 的乘法次序来自当前模型非线性的位置. 

定义 output logit 的局部梯度及状态的局部梯度：

$$
e_t=\frac{\partial L_t}{\partial z_t}
=2w_t(\hat y_t-y_t^*)(1-\hat y_t^2),
\qquad
r_t=\left.\frac{\partial L_t}{\partial x_t}\right|_{\text{local}}
=Bc\,e_t.
\tag{4}
$$

令 $p_t = dL/dx_t$ 为包含未来损失的总梯度. 链式法则给出 BPTT：

$$
p_T=r_T,\qquad
p_t=r_t+J_t^\top p_{t+1}
=r_t+\big[(1-\alpha)I_N+\alpha W^\top D_t\big]p_{t+1},
\quad t=T-1,\ldots,1.
\tag{5}
$$

必要时取 $r_0=0$, 继续得到 $p_0=J_0^\top p_1$. 展开式 (5)：

$$
p_t=\sum_{j=t}^{T}
\big(J_t^\top J_{t+1}^\top\cdots J_{j-1}^\top\big)r_j,
\tag{6}
$$

其中 $j=t$ 时乘积为单位矩阵. 这就是共享参数经过全部未来时间步的误差传播. 

暂将 recurrent coupling $W$ 视为独立节点, 并保持读出中的 B 不变. 一次局部扰动给出

$$
dx_{t+1}\big|_{dW}=\alpha D_t(dW)x_t.
$$

累加各时间节点的共享参数贡献：

$$
dL\big|_{dW}
=\sum_{t=0}^{T-1}p_{t+1}^\top\alpha D_t(dW)x_t
=\operatorname{tr}(G^\top dW),
$$

其中

$$
\boxed{G=\frac{\partial L}{\partial W}\Big|_{\text{recurrent node}}
=\sum_{t=0}^{T-1}q_t x_t^\top,
\qquad q_t=\alpha D_t p_{t+1}.}
\tag{7}
$$

每个外积的秩至多为 1, 但它们的和不必 rank-1, 也不必 rank-K. 单 trial 只有 $\operatorname{rank}(G) \le \min(N,T)$ 这一直接求和上界；batch 求和还会增加外积数. BPTT 并不自动把更新限制在当前 W 的行空间或列空间内. 

## 4. 通过因子参数化应用链式法则

因为

$$
dW=(dA)B^\top+A(dB)^\top,
$$

所以

$$
\begin{aligned}
dL\big|_{\rm recurrent}
&=\operatorname{tr}\big(G^\top(dA)B^\top\big)
+\operatorname{tr}\big(G^\top A(dB)^\top\big)\\
&=\operatorname{tr}\big((GB)^\top dA\big)
+\operatorname{tr}\big((G^\top A)^\top dB\big).
\end{aligned}
$$

因此 recurrent 路径贡献为

$$
\nabla_A L\big|_{\rm recurrent}=GB,
\qquad
\nabla_B L\big|_{\rm recurrent}=G^\top A.
\tag{8}
$$

**当前实现的 B 还直接进入读出, 不能漏掉这条路径. ** 固定状态 $x_t$ 时, 由 $z_t = c^\top B^\top x_t + d$ 得

$$
dL\big|_{B,\rm readout}=\sum_{t=1}^T e_t c^\top(dB)^\top x_t.
$$

当前模型的数据损失完整梯度为

$$
\boxed{
\nabla_A L=GB,\qquad
\nabla_B L=G^\top A+\sum_{t=1}^T e_t x_t c^\top.}
\tag{9}
$$

状态对 B 的依赖已由 BPTT 和式 (8) 计入；式 (9) 的附加项是读出的显式依赖, 没有重复计算. 若读出是独立参数 $v^\top x_t$, 且没有其他显式因子依赖, 才可以把 $\nabla_B L = G^\top A$ 当作完整梯度. 

若采用论文常见的 $W = AB^\top/N$, 式 (8) 两项分别变为 $GB/N$ 和 $G^\top A/N$. 归一化不改变秩界. 若存在因子正则项, 则另加其直接梯度. 

## 5. Adam 更新与秩上界的严格证明

对任一因子 $\Theta \in \{A,B\}$, 其形状始终为 $N\times K$. 令 $g_s^\Theta$ 是第 s 步实际送入 Adam 的梯度, 以标准 Adam 为例：

$$
\begin{aligned}
\mu_s^\Theta&=\beta_1\mu_{s-1}^\Theta+(1-\beta_1)g_s^\Theta,\\
\nu_s^\Theta&=\beta_2\nu_{s-1}^\Theta+(1-\beta_2)(g_s^\Theta\odot g_s^\Theta),\\
\widehat\mu_s^\Theta&=\mu_s^\Theta/(1-\beta_1^s),\qquad
\widehat\nu_s^\Theta=\nu_s^\Theta/(1-\beta_2^s),\\
\Theta_s&=\Theta_{s-1}-\eta_s
\frac{\widehat\mu_s^\Theta}{\sqrt{\widehat\nu_s^\Theta}+\epsilon}.
\end{aligned}
\tag{10}
$$

乘方、开根号和除法逐元素进行, 来自 [Kingma & Ba, Adam, Algorithm 1](https://arxiv.org/abs/1412.6980). 这些运算改变元素, 不改变参数矩阵的形状. 当前 trainer 中的 norm clipping 只缩放梯度；若配置 coupled weight decay, 则 Adam 使用的梯度还含因子的 $\lambda\Theta$ 项. 这些都不影响下面的证明. 

**命题. ** 设 $A_0, B_0 \in \mathbb{R}^{N\times K}$. 如果每一步都更新这两个固定形状的因子, 并且实际 recurrent coupling 始终由 $W_s = A_s B_s^\top$ 定义, 那么对任意参数有限且更新有定义的优化步 s, 

$$
\boxed{\operatorname{rank}(W_s)\le K.}
\tag{11}
$$

**证明. ** 初始时因子各有 K 列. 式 (10) 保持形状, 归纳可知每一步 $A_s, B_s$ 都属于 $\mathbb{R}^{N\times K}$. 对任意 $v \in \mathbb{R}^N$, 

$$
W_sv=A_s(B_s^\top v)\in\operatorname{col}(A_s).
$$

所以 $\operatorname{col}(W_s) \subseteq \operatorname{col}(A_s)$, 从而

$$
\operatorname{rank}(W_s)
\le\operatorname{rank}(A_s)\le K.
$$

同样可得 $\operatorname{rank}(W_s) \le \operatorname{rank}(B_s)$. 证毕. 

此证明对有限学习率精确成立, 不需要把 Adam 近似成连续梯度流；甚至不依赖更新方向是否为精确梯度. 它只保证秩上界, 不保证秩恰好为 K、损失收敛、训练稳定或因子保持高斯分布. 

## 6. 为什么不能把它理解为直接更新 W

若把 Adam 对因子的增量记为 $\Delta A, \Delta B$, 实际连接增量满足精确恒等式

$$
W_{s+1}
=W_s+\Delta A B_s^\top+A_s\Delta B^\top+\Delta A\Delta B^\top
=(A_s+\Delta A)(B_s+\Delta B)^\top.
\tag{12}
$$

最后一个交叉项在有限步长下一般不为零. 保留完整乘积才能得到真实更新；仅保留一阶项通常不保证更新后矩阵仍在 rank-K 集合内. 矩阵 $W_{s+1}-W_s$ 本身也不必 rank-K. 

即使使用普通 SGD, 且假设损失只通过 W 依赖 A、B, 也有

$$
W_{s+1}=W_s-\eta(GB_sB_s^\top+A_sA_s^\top G)
+\eta^2GB_sA_s^\top G.
\tag{13}
$$

因此这不是 $W_{s+1}=W_s-\eta G$. 该式仅用于解释因子更新几何；当前共享 B 的读出还需要式 (9) 的附加贡献. 

**直接对低秩初始化的 W 使用 Adam 的反例. ** 取 $N=2, K=1, \alpha=1$, 一步 tanh RNN、无输入和偏置、固定初态 $x_0=e_2$, 独立恒等读出, 损失为

$$
W_0=\begin{pmatrix}1&0\\0&0\end{pmatrix},\qquad
x_1=\tanh(W_0e_2)=0,\qquad
L=\tfrac12\|x_1-e_2\|^2.
$$

由式 (7), 在该点 $G=-e_2e_2^\top$. 从零一阶、二阶 moment 开始, 第一次 Adam 的 bias correction 给出 $\hat{\mu}=G, \hat{\nu}=G\odot G$, 因此对任意 $\eta>0, \epsilon>0$：

$$
W_1=\begin{pmatrix}1&0\\0&\eta/(1+\epsilon)\end{pmatrix},
\qquad \operatorname{rank}(W_1)=2>K.
\tag{14}
$$

这已经是一个真实 RNN 损失产生的梯度, 而且只用一步 BPTT；所以“BPTT + Adam 自动保持低秩初始化”是错误命题. 该反例读出与本项目不同, 目的是反驳不带因子参数化条件的一般性说法. 

## 7. 当前代码核对与数值验证

- [rank2_ctrnn.py](../src/slow_manifold/models/rank2_ctrnn.py)：`__init__` 注册 `m,n` 两个 `state_size×rank` Parameter；`recurrent_matrix` 返回 `m @ n.T`；`flow` 实际计算 `(state @ n) @ m.T`；`readout` 使用 `state @ n`. 没有独立可训练的 dense W. 当前配置校验限定 `rank=2`, 本文的证明推广至一般 K. 
- [trainer.py](../src/slow_manifold/training/trainer.py)：`train_model` 对 `model.parameters()` 建立 Adam；`_measure_step` 依次执行 rollout、masked MSE、backward、可选 gradient clipping、optimizer.step. 

2026-09-09 对当前公共模型做了补充数值核对：CPU, PyTorch `2.14.0+cu130`, float64, seeds `7,19,41`；N=5、K=2、batch=2、T=7、dt=0.13、tau=0.7, 其余模型字段来自 resolved `experiments/phase1_rank2_baseline.yaml`. 每个 seed 使用独立 torch Generator, 依次初始化模型并生成标准正态 inputs / 非零 initial states / targets 及 uniform masks；第 3 个时间点 mask 置零后整体归一化. 手写式 (3)–(9) 的反向递推与公共 rollout 的 autograd 相比, 两个因子梯度的最大逐元素绝对误差均不超过 $1.12\times 10^{-16}$. 

省略读出直接项会漏掉 Frobenius 范数分别约为 $1.069, 0.776, 0.251$ 的 B 梯度, 确认这一项在当前模型中确实必要. 随后每个 seed 使用 Adam（lr=0.03、weight_decay=0.01、其余默认值）、clip_norm=0.1 连续更新 10 步, 每步数值秩均为 2. 式 (14) 的 dense Adam 反例也已用 autograd 验证：lr=0.1、eps=10⁻⁸ 时新增对角元约为 $0.099999999$, 秩升至 2. 

这些短小核对用于检查推导和代码的一致性, 不是训练效果实验, 也不替代式 (11) 的证明. 临时数值明细位于 `temp/rank-proof/verification.json`, 本文记录足够解释其含义的条件与结果. 

数学证明采用精确实数运算. 浮点矩阵乘法得到的显式 W 可能出现舍入量级的额外小奇异值, 数值秩应使用与 dtype、矩阵尺度相适应的容差；不能要求所有尾部奇异值在浮点数中逐位等于零. 

## 8. 本问题的结论边界

低秩连接不等于完整状态位于 K 维线性子空间, 也不等于系统 Jacobian 的秩为 K. 例如当前模型的 full-state Jacobian 为 $(-I + D W)/\tau$, 一般仍为满秩. 固定 checkpoint 内, $\kappa=B^\top x$ 的闭合方程由乘上 Bᵀ直接得到, 但它的存在本身不证明 slow manifold 或 ghost mechanism. 

对导师问题的准确表述是：**我们把 recurrent coupling 硬参数化为两个 N×K 因子的乘积. BPTT 通过链式法则计算所有因子梯度, 包括共享读出路径；Adam 在因子空间更新, 而每一步的连接始终重新由因子乘积定义. 因此连接的秩始终至多为 K. 只对 dense W 做低秩初始化, 没有这一保证. **
