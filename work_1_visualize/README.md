# Work 1: 数据可视化分析

## 📋 任务概述

本任务对猫狗语音数据集进行全面的数据可视化分析，包括基础统计、波形展示、频谱分析、特征分布、相关性分析和降维可视化等，为后续机器学习和深度学习分类任务提供数据洞察。

## 🔧 环境要求

### Python版本
- Python 3.8+

### 核心依赖库
```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
librosa>=0.9.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### 安装依赖
```bash
pip install numpy pandas matplotlib seaborn librosa scipy scikit-learn
```

## 📁 数据结构

### 输入数据（来自 `../Data/cats_dogs/`）
```
Data/
└── cats_dogs/
    ├── train/
    │   ├── cat/          # 训练集猫类音频 (约125个)
    │   └── dog/          # 训练集狗类音频 (约85个)
    └── test/
        ├── cats/         # 测试集猫类音频 (约39个)
        └── test/         # 测试集狗类音频 (约28个)
```

**数据规模**：
- 总样本数：277个音频文件
- 训练集：210个样本（猫125 + 狗85）
- 测试集：67个样本（猫39 + 狗28）
- 音频格式：WAV文件
- 采样率：16000 Hz

## 🚀 运行指南

### 方式一：运行完整可视化流程

#### 1. 基础数据可视化
```bash
cd work_1_visualize
python visualize_audio.py
```

**功能说明**：
- 自动扫描Data目录下的音频文件
- 生成10张核心可视化图表
- 包含数据统计、波形、频谱、MFCC、特征分布等

**运行时间**：约2-5分钟（取决于CPU性能）

#### 2. 深度特征分析
```bash
python deep_feature_analysis.py
```

**功能说明**：
- 特征相关性矩阵分析
- PCA降维可视化（2D/3D）
- t-SNE非线性降维
- 特征区分度评估（t检验）

**依赖**：需要先运行`visualize_audio.py`生成特征数据

**运行时间**：约3-8分钟

### 方式二：单独运行特定可视化模块

如果只需要生成部分图表，可以修改`visualize_audio.py`中的main函数，注释掉不需要的部分：

```python
if __name__ == '__main__':
    audio_files = get_audio_files()
    
    # 取消注释需要运行的模块
    
    plot_data_statistics(audio_files)           # 图1: 数据统计
    plot_waveform_samples(audio_files)          # 图2: 波形样本
    plot_spectrogram_samples(audio_files)       # 图3: 频谱图
    plot_mfcc_samples(audio_files)              # 图4: MFCC特征
    plot_feature_distribution(audio_files)      # 图5: 特征分布
    # ... 其他函数
```

## 📊 输出内容

### 生成的图表文件（PNG & PDF格式）

所有图表同时保存为PNG（用于预览）和PDF（用于LaTeX报告）两种格式：

| 序号 | 文件名 | 内容描述 | 尺寸 | 用途 |
|------|--------|----------|------|------|
| 1 | `01_data_statistics` | 数据集统计概览（样本数、时长分布、类别比例） | 14×10英寸 | 了解数据基本构成 |
| 2 | `02_waveform_samples` | 典型猫狗叫声波形对比 | 16×12英寸 | 观察时域波形差异 |
| 3 | `03_spectrogram_samples` | 梅尔频谱图对比 | 16×12英寸 | 分析频域特征差异 |
| 4 | `04_mfcc_samples` | MFCC系数热力图 | 16×12英寸 | 展示声学特征模式 |
| 5 | `05_feature_distribution` | 128维特征分布直方图 | 18×14英寸 | 检查特征分布特性 |
| 6 | `06_feature_correlation` | 特征相关性热力图 | 12×10英寸 | 识别冗余特征 |
| 7 | `07_pca_2d` | PCA二维降维散点图 | 10×8英寸 | 线性可分性评估 |
| 8 | `08_pca_3d` | PCA三维降维散点图 | 12×10英寸 | 三维空间聚类观察 |
| 9 | `09_tsne` | t-SNE非线性降维 | 10×8英寸 | 复杂结构发现 |
| 10 | `10_feature_discrimination` | 特征区分度排序 | 14×10英寸 | 筛选关键特征 |

### 生成的数据文件

| 文件名 | 格式 | 内容描述 |
|--------|------|----------|
| `feature_discrimination.csv` | CSV | Top特征区分度统计（t值、p值、效应量） |

## ⚠️ 注意事项

1. **首次运行**需要下载字体（SimHei）或确保系统已安装中文字体
2. **内存需求**：处理277个音频文件约需500MB-1GB RAM
3. **运行路径**：必须在项目根目录下运行，或修改DATA_DIR路径
4. **图形后端**：如无GUI环境，使用`matplotlib.use('Agg')`保存图片而不显示
5. **并行加速**：波形加载可使用多进程（joblib）提速

## 📚 相关文件

- [`visualize_audio.py`](visualize_audio.py) - 基础可视化脚本
- [`deep_feature_analysis.py`](deep_feature_analysis.py) - 高级特征分析脚本

---

**最后更新**: 2026-06-22  
