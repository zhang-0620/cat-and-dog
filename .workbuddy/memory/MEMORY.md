# 项目约定 / Project Conventions

## 猫狗识别项目（E:/cat and dog）

### 运行方式
- Python venv：`C:/Users/asus/.workbuddy/binaries/python/envs/default/Scripts/python.exe`
- 所有命令需设置 `PYTHONPATH="E:/cat and dog/src"`，并以 `python -m catdog.<entry>` 方式运行。
- 入口：`catdog.train` / `catdog.evaluate` / `catdog.predict` / `catdog.report`

### 代码约定
- **零第三方 ML 框架**：算法全部手写（numpy），不引入 PyTorch/TensorFlow/scikit-learn。
- 配置集中在 `configs/config.yaml`，代码不得硬编码超参数。
- 标签约定：cat=0，dog=1；评估时以 dog 为正类计算 P/R/F1。
- 特征提取必须在训练集上 fit 标准化器/PCA，测试集只 transform（防数据泄漏）。
- 可视化统一走 `utils/visualize.py`，中文字体用 Microsoft YaHei / SimHei，`axes.unicode_minus=False`。

### 测试集
- `ceshi/` 为指定的测试集文件夹（100 张，文件名前缀即标签）。
- 当前为单文件夹演示模式（从 ceshi 分层切分训练/验证）；若有独立训练集，配置 `dataset.train_dir` 即可切换。

### 踩坑备忘（勿重犯）
- SVM 前必须做特征标准化，否则 SMO 数值不稳定。
- numpy float 图像传给 matplotlib imshow 必须显式 vmin/vmax，否则全黑。
- MaxPool 用 im2col 实现以兼容非整除尺寸。
