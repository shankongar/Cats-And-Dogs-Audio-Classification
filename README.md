# 🐱🐶 Cat and Dog Audio Classification (猫狗语音分类)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch Version" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.0%2B-green.svg" alt="Scikit-learn Version" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" />
</p>

## 📖 项目简介

本项目是一个完整的**猫狗语音二分类系统**，系统性地探索了从数据可视化、传统机器学习到深度神经网络的全流程音频分类方法。项目包含三个核心任务模块和一个创新模型设计：

- **Work 1**: 数据可视化与特征分析
- **Work 2**: 传统机器学习方法对比（9种算法）
- **Work 3**: 深度学习架构对比（8种网络）
- **self_Improve**: 创新模型MS-TFANet（多尺度时频注意力网络）

### 核心亮点

✨ **最佳性能**: MLP深度学习模型达到 **95.52%** 准确率  
✨ **全面对比**: 17+种模型系统性基准测试  
✨ **可解释性**: SHAP值分析、特征重要性排序  
✨ **学术规范**: 完整的LaTeX实验报告与技术文档  
✨ **工程化**: 一键运行脚本、模型持久化、完整日志  

---

## 🎯 项目结构

```
SpeechClassfier/
│
├── 📁 Data/                          # 原始音频数据
│   └── cats_dogs/                    # 猫狗语音数据集
│       ├── train/                    # 训练集 (210样本)
│       │   ├── cat/                  # 猫类音频 (~125个)
│       │   └── dog/                  # 狗类音频 (~85个)
│       └── test/                     # 测试集 (67样本)
│           ├── cats/                 # 猫类测试样本
│           └── test/                 # 狗类测试样本
│
├── 📁 work_1_visualize/              # 【任务一】数据可视化 ⭐
│   ├── visualize_audio.py            # 基础可视化（10张图表）
│   ├── deep_feature_analysis.py      # 高级特征分析
│   ├── results/                      # 可视化输出（PNG + PDF）
│   │   ├── 01_data_statistics.*      # 数据统计概览
│   │   ├── 02_waveform_samples.*     # 波形可视化
│   │   ├── 03_spectrogram_samples.*  # 频谱图分析
│   │   ├── 04_mfcc_samples.*        # MFCC特征热力图
│   │   ├── 05_feature_distribution.*# 特征分布直方图
│   │   ├── 06_feature_correlation.* # 相关性矩阵
│   │   ├── 07_pca_2d.*              # PCA二维降维
│   │   ├── 08_pca_3d.*              # PCA三维降维
│   │   ├── 09_tsne.*                # t-SNE非线性降维
│   │   └── 10_feature_discrimination.* # 特征区分度评估
│   ├── experiment_report.tex         # LaTeX实验报告
│   ├── experiment_report.html        # HTML报告预览
│   └── README.md                     # 详细使用文档
│
├── 📁 work_2_ML_classfier/           # 【任务二】传统机器学习 ⭐
│   ├── run_all.py                    # 一键运行入口
│   ├── config.py                     # 配置文件
│   ├── feature_extraction.py         # 128维特征提取
│   ├── train_models.py               # 9种ML模型训练
│   ├── evaluate_models.py            # 模型评估与可视化
│   ├── ensemble_learning.py          # 集成学习研究
│   ├── error_analysis.py             # 错误分析与困难样本
│   ├── model_interpretability.py     # SHAP解释性分析
│   ├── model_ml/                     # 训练好的模型权重
│   │   ├── xgboost_best.pkl          # 最佳模型 (94.03%)
│   │   ├── gradient_boosting_best.pkl
│   │   ├── svm_best.pkl
│   │   ├── random_forest_best.pkl
│   │   └── ... (共13个模型文件)
│   ├── results/                      # 实验结果
│   │   ├── figures/                  # 可视化图表 (8张PDF)
│   │   ├── *.npy                     # 特征矩阵缓存
│   │   └── model_comparison.csv      # 性能汇总表
│   ├── experiment_report.tex         # LaTeX实验报告
│   └── README.md                     # 详细使用文档
│
├── 📁 work_3_dl_classfier/           # 【任务三】深度学习 ⭐
│   ├── train_all.py                  # 一键训练所有模型
│   ├── config.py                     # 训练配置
│   ├── data_loader.py                # 数据加载器
│   ├── model_mlp.py                  # 多层感知机 (95.52%) 🏆
│   ├── model_cnn1d.py                # 一维卷积网络 (94.03%)
│   ├── model_rnn.py                  # LSTM循环网络 (67.16%)
│   ├── model_cnn_rnn.py              # CNN-RNN混合 (91.04%)
│   ├── model_mlp_attention.py        # 注意力MLP
│   ├── model_cnn_residual.py         # 残差CNN
│   ├── model_bilstm.py               # 双向LSTM
│   ├── model_gru.py                  # 门控循环单元
│   ├── model_dl/                     # PyTorch模型权重
│   │   ├── mlp_best.pth              # 最佳DL模型
│   │   └── ... (共8个.pth文件)
│   ├── results/                      # 训练结果
│   │   ├── figures/                  # 训练曲线图 (16张)
│   │   ├── logs/                     # 训练日志
│   │   └── *.csv                     # 性能指标
│   ├── experiment_report.tex         # LaTeX实验报告
│   └── README.md                     # 详细使用文档
│
├── 📁 self_Improve/                  # 【创新任务】自定义模型
│   └── (创新架构设计与实现)
│
├── 📄 main.py                        # 项目主入口（统一调度）
├── 📄 convert_png_to_pdf.py          # PNG转PDF工具脚本
├── 📄 README.md                      # 本文件 - 项目总览
├── 📄 LICENSE                        # 开源协议 (MIT)
└── 📄 .gitignore                     # Git忽略规则
```

---

## 🚀 快速开始

### 环境要求

- **操作系统**: Windows / Linux / macOS
- **Python版本**: 3.8 或更高
- **内存建议**: ≥4GB RAM
- **GPU**: 可选（CPU即可运行）

### 安装依赖

#### 方式一：使用pip安装（推荐）

```bash
# 克隆仓库
git@github.com:shankongar/Cats-And-Dogs-Audio-Classification.git
cd Cats-And-Dogs-Audio-Classification

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装核心依赖
pip install numpy pandas matplotlib seaborn scikit-learn librosa scipy joblib

# 安装机器学习扩展库
pip install xgboost hmmlearn shap

# 安装深度学习库（CPU版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或安装GPU版本（如有NVIDIA GPU）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 方式二：使用requirements.txt（如果提供）

```bash
pip install -r requirements.txt
```

### 运行流程

#### 方式一：按顺序执行三个任务（推荐新手）

```bash
# Step 1: 数据可视化分析
cd work_1_visualize
python visualize_audio.py
python deep_feature_analysis.py
cd ..

# Step 2: 传统机器学习分类
cd work_2_ML_classfier
python run_all.py
cd ..

# Step 3: 深度学习分类
cd work_3_dl_classfier
python train_all.py
cd ..
```

#### 方式二：使用统一入口（高级用户）

```bash
python main.py
```

**`main.py`功能**：
- 自动检测依赖和环境
- 按顺序执行Work 1 → Work 2 → Work 3
- 显示进度和运行状态
- 错误处理和日志记录

#### 预计运行时间

| 任务 | CPU环境 | GPU环境 |
|------|---------|---------|
| Work 1: 数据可视化 | 5-10分钟 | 5-10分钟 |
| Work 2: 机器学习训练 | 5-15分钟 | 5-15分钟 |
| Work 3: 深度学习训练 | 15-40分钟 | 5-15分钟 |
| **总计** | **25-65分钟** | **15-40分钟** |

---

## 📊 实验结果总览

### 性能排行榜

| 排名 | 模型 | 类型 | 准确率 | F1分数 | 特点 |
|:---:|------|------|:------:|:------:|------|
| 🥇 | **MLP** | Deep Learning | **95.52%** | **0.9455** | 最简单但最有效！ |
| 🥈 | XGBoost | ML (Boosting) | 94.03% | 0.9259 | 传统ML最强 |
| 🥈 | GBDT | ML (Boosting) | 94.03% | 0.9259 | 与XGBoost并列 |
| 🥉 | CNN1D | DL (CNN) | 94.03% | 0.9259 | 收敛最快(21轮) |
| 4 | SVM | ML (Kernel) | 92.54% | 0.9057 | 小样本稳定性好 |
| 4 | Random Forest | ML (Bagging) | 92.54% | 0.9057 | 可解释性强 |
| 4 | AdaBoost | ML (Boosting) | 92.54% | 0.9057 | 集成方法有效 |
| 6 | Naive Bayes | ML (Generative) | 91.04% | 0.8846 | 极致速度(0.002s) |
| 7 | CNN-RNN | DL (Hybrid) | 91.04% | 0.8929 | 过拟合风险最低 |
| 8 | BiLSTM | DL (RNN) | 89.55% | - | 双向建模 |
| 9 | Logistic Regression | ML (Linear) | 89.55% | 0.8679 | 基线模型 |
| 9 | KNN | ML (Instance) | 89.55% | 0.8727 | 懒学习 |
| ❌ | RNN | DL (RNN) | 67.16% | 0.6667 | 输入不匹配导致失败 |

### 关键发现

1. **没有免费午餐定理**: 在小样本+强特征场景下，简单MLP优于复杂RNN/CNN-RNN
2. **特征工程至关重要**: 128维手工设计特征提供了强大的判别能力
3. **传统ML仍具竞争力**: XGBoost仅比最佳DL低1.49%，且速度更快、更易解释
4. **输入表示决定成败**: RNN失败的根本原因是统计特征不适合序列建模

---

## 🔬 技术栈详情

### 数据处理技术

| 技术 | 用途 | 维度 |
|------|------|------|
| MFCC (Mel-Frequency Cepstral Coefficients) | 声学特征提取 | 78维 |
| Spectral Features (质心/带宽/滚降) | 频域特征 | 28维 |
| Chroma Features | 色度/音高特征 | 24维 |
| Time-Domain Features (ZCR/RMS) | 时域特征 | 4维 |
| StandardScaler | 特征标准化 | 128维总计 |

### 机器学习算法

- **线性模型**: Logistic Regression
- **核方法**: Support Vector Machine (RBF Kernel)
- **实例学习**: K-Nearest Neighbors
- **生成模型**: Gaussian Naive Bayes, HMM
- **Bagging集成**: Random Forest
- **Boosting集成**: GBDT, XGBoost, AdaBoost
- **高级集成**: Voting (Hard/Soft), Stacking

### 深度学习架构

- **全连接网络**: MLP, MLP-Attention
- **卷积网络**: CNN1D, CNN-Residual
- **循环网络**: RNN (LSTM), BiLSTM, GRU
- **混合架构**: CNN-RNN (CNN + BiGRU)

### 训练策略

- **优化器**: Adam (lr=0.0002)
- **损失函数**: CrossEntropyLoss
- **正则化**: Dropout (0.3-0.4), Early Stopping (patience=10)
- **数据划分**: 75.8% train / 24.2% test
- **超参数优化**: Grid Search (ML) / Optuna (DL可选)

---

## 📁 核心模块说明

### Work 1: 数据可视化 (`work_1_visualize/`)

**目标**: 全面理解数据分布和特征特性

**主要功能**:
- ✅ 数据统计分析（样本数、时长、类别平衡）
- ✅ 时域波形可视化
- ✅ 频谱图和梅尔频谱图分析
- ✅ MFCC特征模式展示
- ✅ 128维特征分布检验
- ✅ 特征相关性分析
- ✅ 降维可视化 (PCA/t-SNE)
- ✅ 特征区分度评估 (t-test)

**输出**: 10组高质量图表 (PNG + PDF格式)

📖 [详细文档](work_1_visualize/README.md)

---

### Work 2: 传统机器学习 (`work_2_ML_classfier/`)

**目标**: 系统性对比多种ML算法并建立性能基线

**主要功能**:
- ✅ 128维声学特征提取与缓存
- ✅ 9种ML算法训练与网格搜索调优
- ✅ 5折交叉验证评估
- ✅ 集成学习研究 (Voting/Stacking)
- ✅ 模型解释性分析 (SHAP值)
- ✅ 错误案例诊断与困难样本识别
- ✅ ROC曲线与混淆矩阵可视化

**最佳模型**: XGBoost / GBDT (94.03%)

📖 [详细文档](work_2_ML_classfier/README.md)

---

### Work 3: 深度学习 (`work_3_dl_classfier/`)

**目标**: 探索不同DL架构在小样本场景下的性能表现

**主要功能**:
- ✅ 8种神经网络架构实现
- ✅ 统一训练框架 (Early Stopping + 日志记录)
- ✅ 训练曲线可视化与对比
- ✅ 超参数自动化调优 (Optuna)
- ✅ 过拟合检测与正则化策略
- ✅ 模型权重保存与加载

**最佳模型**: MLP (95.52%)

📖 [详细文档](work_3_dl_classfier/README.md)

---

## 💡 使用示例

### 示例1: 使用训练好的模型进行预测

```python
import joblib
import numpy as np

# 加载XGBoost模型
model = joblib.load('work_2_ML_classfier/model_ml/xgboost_best.pkl')
scaler = joblib.load('work_2_ML_classfier/model_ml/scaler.pkl')

# 准备新音频的特征（128维）
new_features = np.random.randn(1, 128)  # 替换为实际提取的特征
features_scaled = scaler.transform(new_features)

# 预测
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0]

print(f"预测类别: {'猫' if prediction == 0 else '狗'}")
print(f"置信度: {max(probability):.2%}")
```

### 示例2: 使用PyTorch深度学习模型预测

```python
import torch
from work_3_dl_classfier.model_mlp import create_mlp_model
from work_3_dl_classfier.data_loader import load_features

# 加载数据和模型
_, X_test, _, y_test = load_features()
model = create_mlp_model(X_test.shape[1])

checkpoint = torch.load('work_3_dl_classfier/model_dl/mlp_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test))
    predictions = torch.argmax(outputs, dim=1)

accuracy = (predictions == torch.LongTensor(y_test)).float().mean()
print(f"测试准确率: {accuracy:.4f}")
```

### 示例3: 自定义新模型并训练

```python
# 在 work_3_dl_classfier/ 中创建 model_custom.py
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def create_custom_model(input_dim):
    return CustomModel(input_dim)

# 在 train_all.py 中添加调用即可自动训练
```

---

## 🔧 配置与定制

### 修改特征提取参数

编辑 [`work_2_ML_classfier/config.py`](work_2_ML_classfier/config.py):

```python
FEATURE_PARAMS = {
    'sr': 16000,           # 采样率 (Hz)
    'n_mfcc': 13,          # MFCC系数数量
    'n_fft': 2048,         # FFT窗口大小
    'hop_length': 512,     # 跳跃长度
}
```

### 修改训练超参数

编辑 [`work_3_dl_classfier/config.py`](work_3_dl_classfier/config.py):

```python
TRAINING_PARAMS = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'early_stopping_patience': 10,
}
```

### 添加新的机器学习模型

在 [`work_2_ML_classfier/train_models.py`](work_2_ML_classfier/train_models.py) 中参照现有代码添加新模型类。

### 添加新的深度学习架构

1. 在 `work_3_dl_classfier/` 中创建 `model_new.py`
2. 定义模型类和工厂函数 `create_new_model(input_dim)`
3. 在 [`train_all.py`](work_3_dl_classfier/train_all.py) 的 `main()` 函数中添加调用

---

## 📈 项目统计

### 代码规模

- **Python源文件**: ~30个
- **总代码行数**: ~8000+ 行
- **注释覆盖率**: >30%
- **支持的模型数量**: 17+

### 数据规模

- **音频样本总数**: 277个WAV文件
- **训练集**: 210个样本 (猫125 + 狗85)
- **测试集**: 67个样本 (猫39 + 狗28)
- **特征维度**: 128维
- **数据格式**: WAV (16kHz, 单声道)

### 实验产出

- **可视化图表**: 34+ 张 (PNG + PDF双格式)
- **训练好的模型**: 21个 (.pkl / .pth)
- **CSV数据表**: 15+
- **日志文件**: 11个

---

### Issue模板

报告Bug时请包含：
- 操作系统和Python版本
- 完整的错误信息（Traceback）
- 复现步骤
- 期望行为 vs 实际行为

---

## 📄 许可证

本项目采用 **MIT License** 开源协议。详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2026 Speech Classification Project Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👥 作者团队

**Speech Classification Project Team**

- **项目负责人**: [shankongar]
- **核心开发**: [shankongar]

---

## 📞 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/shankongar/SpeechClassfier/issues)
- **技术讨论**: [Discussions](https://github.com/shankongar/SpeechClassfier/discussions)
- **邮箱**: [float_inf@foxmail.com]

---

## ⭐ Star历史

如果这个项目对您有帮助，请给一个 ⭐ 支持一下！

<a href="https://github.com/YOUR_USERNAME/SpeechClassfier/stargazers">
  <img src="https://img.shields.io/github/stars/YOUR_USERNAME/SpeechClassfier?style=social" alt="Stars" />
</a>

---

<p align="center">
  <b> Made with ❤️ by Speech Classification Project Team </b>
</p>

<p align="center">
  <sub>最后更新: 2026-06-22 | 版本: v2.0 | 许可证: MIT</sub>
</p>
