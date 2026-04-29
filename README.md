# 物理信息神经网络 (PINN) 改进算法实验汇总

本仓库包含了关于 **Physics-Informed Neural Networks (PINNs)** 求解各种经典偏微分方程（PDEs）的一系列改进算法与对比实验。核心目标是探索不同的采样策略与边界条件处理方式，以提升模型在复杂物理场（如激波、相变、孤子波）中的逼近精度和收敛速度。

##  核心算法与实验步骤
项目中的三个子目录分别对应三个经典的偏微分方程，每个目录下都遵循了统一的实验推进逻辑（编号 1-4）：

- **[1] Baseline PINN**: 标准的物理信息神经网络。
- **[2] AB-PINN (Adaptive / Hard Boundary)**: 改进了边界条件处理方式（如引入硬约束或自适应权重）的 PINN。
- **[3] RAD / RADS-PINN**: 基于残差的自适应采样 (Residual-based Adaptive Sampling) 策略，优化配点分布。
- **[4] RADS + AB-PINN**: 结合了自适应采样与改进边界处理的最终融合模型。

##  项目结构说明

### 1. `Allen cahn/` (Allen-Cahn 方程)
用于模拟相场动力学问题。
- 包含从 `1PINN` 到 `4RADS+AB` 的完整训练 (`train_*.py`) 与可视化 (`plot_*.py`) 脚本。
- **`ablation/`**: 存放了针对 Allen-Cahn 方程的消融实验代码，用于单独验证各模块的有效性。
- 依赖数据：`AC.mat`

### 2. `burgers/` (Burgers' 方程)
用于研究非线性激波（Shock Wave）的捕捉与拟合。
- 包含了针对 Burgers 方程的动态自适应边界与残差采样实验 (`train_rad_dynamic_ab_pinn.py`)。
- 依赖数据：`burgers_shock.mat`

### 3. `KDV/` (Korteweg-de Vries 方程)
用于模拟浅水波等孤立波（Soliton）的演化。
- 包含了针对 KdV 方程的标准 PINN、AB-PINN、RAD-PINN 及其组合方法的实验脚本。
- 依赖数据：`KdV.mat`

##  环境依赖

推荐使用以下环境运行本项目：
- Python 3.8+
- PyTorch (建议支持 GPU 以加速训练)
- NumPy
- SciPy (用于读取 `.mat` 数据)
- Matplotlib (用于结果可视化)


##  数据与结果
- 每个目录下均提供了参考用的精确解/数值解数据（如 .mat 文件）。

- 运行可视化脚本后，生成的误差热力图、对比曲线等会保存为 .pdf 格式（如 Figure 1.pdf, Figure 2.pdf 等）

