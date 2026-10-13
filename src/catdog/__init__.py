"""Cat-Dog Recognition - 猫狗识别计算机视觉项目.

模块结构:
    catdog.data       - 数据集加载与预处理
    catdog.features   - 特征提取 (HOG / 颜色直方图 / PCA)
    catdog.models     - 分类器 (SVM / CNN)
    catdog.utils      - 评估指标与可视化工具
"""

__version__ = "1.0.0"
__all__ = ["data", "features", "models", "utils"]
