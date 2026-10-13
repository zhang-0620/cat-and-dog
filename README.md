# 猫狗识别 · 计算机视觉项目 / Cat-Dog Recognition

一个**从零实现**的猫狗二分类计算机视觉项目。除 `numpy` / `pillow` / `matplotlib` 外
不使用任何深度学习框架（无 PyTorch / TensorFlow / scikit-learn），所有算法
—— HOG 特征、PCA 降维、SMO 求解的核 SVM、CNN 的前向与反向传播 —— 均为手写实现。

> **测试集**：`E:/cat and dog/ceshi`（100 张，cat 50 + dog 50）

---

## 1. 项目结构

```
E:/cat and dog/
├── ceshi/                        # ★ 测试集（100 张，cat.*.jpg / dog.*.jpg）
├── configs/
│   └── config.yaml               # 全局配置：数据、特征、模型、评估
├── src/catdog/
│   ├── data/
│   │   └── dataset.py            # 数据集扫描、图像读取、预处理、分层切分
│   ├── features/
│   │   ├── hog.py                # HOG 方向梯度直方图（纯 NumPy）
│   │   ├── color_hist.py         # RGB/HSV 颜色直方图
│   │   ├── pca.py                # PCA 主成分分析（SVD 实现）
│   │   └── pipeline.py           # 特征流水线 + 标准化器
│   ├── models/
│   │   ├── svm.py                # 核 SVM（SMO + Platt 概率校准）
│   │   ├── logistic.py           # 逻辑回归基线
│   │   └── cnn.py                # CNN（Conv/BN/ReLU/Pool/Dense 手写反向传播）
│   ├── utils/
│   │   ├── metrics.py            # 混淆矩阵、P/R/F1、ROC-AUC
│   │   ├── visualize.py          # 混淆矩阵图、ROC 曲线、样例网格
│   │   └── io.py                 # 配置加载、JSON 落盘
│   ├── train.py                  # 训练入口
│   ├── evaluate.py               # 测试集评估入口
│   └── predict.py                # 单图 / 批量预测
├── models/                       # 训练好的模型权重（.npz）
├── outputs/                      # 评估结果：指标 JSON、图表 PNG、预测明细 CSV
└── README.md
```

## 2. 环境依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥ 3.9 | 运行环境 |
| numpy | ≥ 1.20 | 全部数值计算 |
| pillow | ≥ 8.0 | 图像读取与缩放 |
| matplotlib | ≥ 3.3 | 可视化输出 |
| pyyaml | 可选 | 解析 YAML 配置（缺失时用内置解析器） |

安装：

```bash
pip install numpy pillow matplotlib pyyaml
```

## 3. 快速开始

所有命令在项目根目录 `E:/cat and dog` 下执行，并设置 `PYTHONPATH` 指向 `src`：

```bash
# Windows (Git Bash)
export PYTHONPATH="E:/cat and dog/src"

# 在测试集 ceshi 上评估全部模型（SVM / 逻辑回归 / CNN）
python -m catdog.evaluate

# 只跑传统特征模型（快速，几秒）
python -m catdog.evaluate --models svm logreg

# 单独训练（按比例切分验证集）
python -m catdog.train --data-dir ceshi --model svm

# 单图预测
python -m catdog.predict --image ceshi/cat.1.jpg --model svm
```

## 4. 技术方案

### 4.1 数据管道
- **扫描**：按文件名前缀识别类别（`cat.*` → 0，`dog.*` → 1），兼容扁平与子目录两种布局。
- **预处理**：统一缩放至 `96×96` RGB，像素归一化到 `[0, 1]`。
- **训练时增强**：随机水平翻转。
- **划分**：按类别**分层切分**，保证训练/验证集中猫狗比例一致。

### 4.2 特征工程（传统方法路线）
| 特征 | 维度 | 说明 |
|------|------|------|
| HOG | 4356 | 9 方向 bin，8×8 cell，2×2 block，L2-Hys 归一化 —— 刻画轮廓形状 |
| 颜色直方图 | 48 | HSV 三通道各 16 bin —— 刻画毛色分布 |
| 合计 | 4404 | 拼接后经 **z-score 标准化** |
| PCA | 79 | 保留 100% 累计方差，降维去噪 |

> `pipeline.py` 中的 `StandardScaler` 是必需的：HOG（0~0.2 量级）与颜色直方图
> （0~0.2 但分布不同）量纲差异会导致核 SVM 的 SMO 数值不稳定。

### 4.3 模型

**核 SVM（`models/svm.py`）**
- 支持 `linear` / `rbf` / `poly` 三种核，二分类标签映射到 ±1。
- **SMO** 求解对偶问题，含 KKT 违背检查 + 第二变量启发式选择（`argmax |Ei−Ej|`）+ 容差逐步收紧。
- **Platt 缩放**：以牛顿法拟合 `p(dog) = σ(A·f + B)`，为 ROC-AUC 提供概率输出。

**CNN（`models/cnn.py`）**
- 结构：`Conv-BN-ReLU-Pool ×3 → Flatten → Dense → Dropout → Dense(1)+Sigmoid`
- 卷积用 **im2col** 加速，池化用 im2col+argmax，反向传播全部手写。
- 含 BatchNorm2D、Dropout、He 初始化、余弦学习率衰减。
- 已通过**数值梯度校验**（卷积 dx / dW、池化 dx 的相对误差均在 1e-3 量级）。

**逻辑回归（`models/logistic.py`）** —— 最简对照基线，含特征标准化。

### 4.4 评估
- 指标：Accuracy / Precision / Recall / F1 / ROC-AUC（均为纯 NumPy 实现）。
- 输出：混淆矩阵（原始 + 归一化）、ROC 曲线、预测样例网格、指标对比柱状图、
  逐样本预测明细 CSV。

## 5. 测试集设定

测试集为 `ceshi/` 目录，共 100 张（cat 50 + dog 50），**文件名即标签**，无需标注文件。

由于项目仅提供该单一文件夹，`evaluate.py` 采用**演示模式**：
从 `ceshi` 中分层切出 80 张作为训练集、20 张作为验证，再在全部 100 张上测试。

> ⚠ **严格实验规范提示**：真实实验中训练集与测试集必须物理隔离，
> 否则测试指标会偏乐观。若后续补充独立训练集，在 `config.yaml` 中设置
> `dataset.train_dir` 即可自动切换为「独立训练集 + ceshi 测试集」模式。

## 6. 输出产物（`outputs/`）

| 文件 | 内容 |
|------|------|
| `test_results.json` | 各模型完整指标 + 混淆矩阵 + 逐类别指标 |
| `cm_<model>.png` / `cm_<model>_norm.png` | 混淆矩阵（原始 / 归一化） |
| `roc_<model>.png` | ROC 曲线 |
| `samples_<model>.png` | 预测样例网格（绿=正确、红=错误） |
| `metrics_comparison.png` | 模型指标对比柱状图 |
| `preprocess_grayscale.png` | 预处理（灰度化 + 缩放）效果 |
| `predictions_<model>.csv` | 逐样本预测明细 |

## 7. 配置说明（`configs/config.yaml`）

关键字段：

```yaml
dataset:
  test_dir: "ceshi"          # ★ 测试集目录
  train_dir: ""              # 独立训练集（留空则用测试集分层切分）
features:
  hog: { enabled: true, orientations: 9, pixels_per_cell: [8, 8] }
  color_hist: { enabled: true, bins: 16, space: "HSV" }
  pca: { enabled: true, n_components: 120 }
models:
  svm: { kernel: "rbf", C: 2.0, gamma: "scale" }
  cnn: { filters: [16, 32, 64], epochs: 60, learning_rate: 0.002, dropout: 0.5 }
```
