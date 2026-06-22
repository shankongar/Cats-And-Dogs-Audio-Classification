# Work 2: 基于传统机器学习的猫狗语音分类

## 📋 任务概述

本任务使用多种传统机器学习算法对猫狗语音数据进行分类，包括逻辑回归、支持向量机、K近邻、朴素贝叶斯、随机森林、梯度提升树、XGBoost、AdaBoost以及隐马尔可夫模型（HMM），并通过网格搜索进行超参数优化，最终对比各模型性能并保存最佳模型。

## 🔧 环境要求

### Python版本
- Python 3.8+

### 核心依赖库
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
librosa>=0.9.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
hmmlearn>=0.2.7
shap>=0.40.0  # 可选：用于模型解释性分析
```

### 安装依赖
```bash
pip install numpy pandas scikit-learn xgboost librosa scipy matplotlib seaborn joblib hmmlearn shap
```

## 📁 数据与文件结构

### 输入数据来源
- **音频数据**: `../Data/cats_dogs/` (WAV格式)
- **特征数据**: 自动提取并缓存至 `results/` 目录

### 项目结构
```
work_2_ML_classfier/
├── config.py                    # 配置文件（特征参数、模型参数）
├── feature_extraction.py        # 特征提取模块（128维声学特征）
├── feature_extraction_hmm.py    # HMM专用特征提取
├── train_models.py              # 模型训练主程序（9种ML算法）
├── evaluate_models.py           # 模型评估与可视化
├── ensemble_learning.py         # 集成学习研究（Voting/Stacking）
├── error_analysis.py            # 错误分析与困难样本识别
├── model_interpretability.py    # 模型解释性分析（SHAP值）
├── model_hmm.py                 # 隐马尔可夫模型实现
├── run_all.py                   # 一键运行所有流程 ⭐
│
├── model_ml/                    # 训练好的模型保存目录
│   ├── logistic_regression_best.pkl
│   ├── svm_best.pkl
│   ├── knn_best.pkl
│   ├── naive_bayes_best.pkl
│   ├── random_forest_best.pkl
│   ├── gradient_boosting_best.pkl
│   ├── xgboost_best.pkl
│   ├── adaboost_best.pkl
│   ├── hmm_best.pkl
│   ├── voting_hard.pkl          # 硬投票集成
│   ├── voting_soft.pkl          # 软投票集成
│   ├── stacking.pkl             # Stacking集成
│   └── scaler.pkl               # 特征标准化器
│
├── results/                     # 实验结果输出目录
    ├── figures/                 # 可视化图表（PNG + PDF）
    │   ├── 01_model_comparison.png/pdf      # 模型性能对比柱状图
    │   ├── 02_confusion_matrices.png/pdf     # 混淆矩阵热力图
    │   ├── 03_roc_curves.png/pdf            # ROC曲线对比
    │   ├── 04_feature_importance.png/pdf     # 特征重要性排序
    │   ├── ensemble_improvement.png/pdf      # 集成学习提升效果
    │   ├── ensemble_diversity.png/pdf        # 集成多样性分析
    │   ├── error_distribution.png/pdf        # 错误类型分布
    │   └── sample_difficulty.png/pdf         # 困难样本识别
    ├── X_train.npy               # 训练集特征矩阵
    ├── X_test.npy                # 测试集特征矩阵
    ├── y_train.npy               # 训练集标签
    ├── y_test.npy                # 测试集标签
    ├── model_comparison.csv      # 模型性能汇总表
    └── error_analysis_report.txt # 错误分析报告

```

## 🚀 运行指南

### 方式一：一键运行完整流程（推荐）

```bash
cd work_2_ML_classfier
python run_all.py
```

**执行流程**：
1. **步骤1**: 导入模块和依赖库
2. **步骤2**: 特征提取 + 模型训练（9种算法）
3. **步骤3**: 加载训练数据
4. **步骤4**: 模型评估与可视化生成

**运行时间**：约5-15分钟（取决于CPU性能）

**控制台输出示例**：
```
======================================================================
开始机器学习分类任务
======================================================================

步骤1: 导入模块...

步骤2: 训练模型...
正在提取128维声学特征...
  [====================] 100% (277/277) [00:45<00:00]

开始网格搜索超参数优化...
  训练 Logistic Regression...
  训练 SVM...
  训练 KNN...
  ... (共9个模型)

步骤3: 加载训练数据...
步骤4: 评估模型...
  生成混淆矩阵...
  绘制ROC曲线...
  分析特征重要性...
  ...

======================================================================
任务完成！
======================================================================
```

---

### 方式二：分步运行（适合调试）

#### Step 1: 特征提取
```bash
python -c "from feature_extraction import main; main()"
```

**功能**：
- 从WAV音频提取128维特征
- 特征组成：MFCC(78维) + 时域(4维) + 频域(28维) + 色度(24维)
- StandardScaler标准化处理
- 保存为NumPy数组格式（`.npy`）

**输出文件**：
- `results/X_train.npy` - 训练集特征 (210, 128)
- `results/X_test.npy` - 测试集特征 (67, 128)
- `results/y_train.npy` - 训练集标签 (210,)
- `results/y_test.npy` - 测试集标签 (67,)
- `model_ml/scaler.pkl` - 标准化器（用于新数据预测）


#### Step 2: 模型训练
```bash
python train_models.py
```

**训练模型列表**：

| 序号 | 模型名称 | 算法类型 | 超参数搜索空间 |
|------|----------|----------|----------------|
| 1 | Logistic Regression | 线性模型 | C∈{0.001-100}, solver∈{lbfgs, liblinear} |
| 2 | SVM | 核方法 | C∈{0.1-10}, kernel∈{rbf, linear}, gamma |
| 3 | KNN | 实例学习 | k∈{3,5,7,9,11}, weights, metric |
| 4 | Naive Bayes | 生成模型 | 无需调参（高斯朴素贝叶斯） |
| 5 | Random Forest | Bagging集成 | n_estimators∈{50,100,200}, max_depth |
| 6 | GBDT | Boosting集成 | n_estimators, learning_rate, max_depth |
| 7 | XGBoost | 梯度提升 | n_estimators, max_depth, learning_rate |
| 8 | AdaBoost | 自适应提升 | n_estimators, learning_rate |
| 9 | HMM | 序列模型 | 隐状态数=5, 协方差类型=对角阵 |

**训练策略**：
- 5折交叉验证（GridSearchCV）
- 评估指标：准确率（scoring='accuracy'）
- 类别权重平衡：`class_weight='balanced'`

**输出文件**：
- `model_ml/{模型名}_best.pkl` - 最佳模型权重
- `results/model_comparison.csv` - 性能指标汇总


#### Step 3: 模型评估与可视化
```bash
python evaluate_models.py
```

**生成的可视化内容**：
- 各模型性能对比柱状图
- 混淆矩阵热力图（XGBoost/SVM/RF）
- ROC曲线及AUC值对比
- 特征重要性Top20排序
- 训练时间 vs 准确率散点图

#### Step 4: 高级分析（可选）
```bash
# 集成学习研究
python ensemble_learning.py

# 错误分析
python error_analysis.py

# SHAP模型解释性分析（需要安装shap库）
python model_interpretability.py
```

---

### 方式三：使用已训练模型进行预测

```python
import joblib
import numpy as np
from feature_extraction import extract_features_single

# 1. 加载模型和标准化器
model = joblib.load('model_ml/xgboost_best.pkl')
scaler = joblib.load('model_ml/scaler.pkl')

# 2. 提取新音频的特征
audio_path = 'test_cat_001.wav'
features = extract_features_single(audio_path)  # 返回(128,)向量

# 3. 标准化
features_scaled = scaler.transform(features.reshape(1, -1))

# 4. 预测
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0]

print(f"预测类别: {'猫' if prediction == 0 else '狗'}")
print(f"置信度: {max(probability):.2%}")
```

## 🔧 配置参数说明

编辑 [`config.py`](config.py) 可自定义以下参数：

```python
# 特征提取参数
FEATURE_PARAMS = {
    'sr': 16000,           # 采样率
    'n_mfcc': 13,          # MFCC系数个数
    'n_fft': 2048,         # FFT窗口大小
    'hop_length': 512,     # 跳跃长度
}

# 模型训练参数
MODEL_PARAMS = {
    'cv_folds': 5,         # 交叉验证折数
    'random_state': 42,    # 随机种子
    'scoring': 'accuracy'  # 评估指标
}
```

## ⚠️ 注意事项

1. **首次运行**会自动从WAV文件提取特征并缓存（约1-2分钟），后续运行直接加载缓存
2. **内存需求**: 特征提取阶段需约500MB RAM，模型训练阶段需约1GB
3. **GPU支持**: 本任务主要使用CPU，无需GPU加速
4. **随机种子**: 已固定为42，确保实验可复现
5. **HMM模型**: 使用hmmlearn库，如遇安装问题可跳过（不影响其他模型）
6. **Windows兼容性**: 在Windows上运行时可能需要设置`multiprocessing_method='loky'`

## 🐛 常见问题解决

### Q1: 找不到Data目录？
确保项目结构正确，或在`feature_extraction.py`中修改`DATA_DIR`路径。

### Q2: 特征提取太慢？
首次提取后会自动缓存到`results/*.npy`，后续运行秒级加载。

### Q3: 内存不足？
减少`config.py`中的`n_mfcc`或降低音频采样率。

### Q4: 某些模型报错？
检查是否安装了对应库（如xgboost, hmmlearn）。可在`train_models.py`中注释掉该模型。

### Q5: 如何添加新模型？
参考现有模型代码，在`train_models.py`中添加新的模型类和超参数空间。

**最后更新**: 2026-06-22  
