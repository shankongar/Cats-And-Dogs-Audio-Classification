# Work 3: 基于深度神经网络的猫狗语音分类

## 📋 任务概述

本任务使用多种深度神经网络架构对猫狗语音数据进行分类，包括MLP（多层感知机）、CNN1D（一维卷积网络）、RNN（循环神经网络）、CNN-RNN混合模型，以及4种变体模型（BiLSTM、GRU、MLP-Attention、CNN-Residual）。通过对比不同架构在小样本场景下的性能，揭示"没有免费午餐定理"的具体体现。

## 🔧 环境要求

### Python版本
- Python 3.8+

### 核心依赖库
```
torch>=2.0.0              # PyTorch深度学习框架
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.64.0              # 进度条显示
optuna>=3.0.0             # 超参数自动优化（可选）
```

### 安装依赖

#### CPU版本
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn tqdm optuna
```

#### GPU版本
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 或根据CUDA版本选择对应链接
```

**说明**: 本任务数据量较小（277样本），CPU训练已足够快速，GPU可进一步提速但非必需。

## 📁 数据与文件结构

### 输入数据来源
- **特征数据**: 复用Work 2提取的128维特征（`../work_2_ML_classfier/results/*.npy`）
- **无需重新提取特征**: 直接加载缓存的特征矩阵

### 项目结构
```
work_3_dl_classfier/
├── config.py                    # 训练配置（超参数、路径等）
├── data_loader.py               # 数据加载器（从Work2加载特征）
├── train_all.py                 # 一键训练所有模型 ⭐
├── evaluate_models.py           # 模型评估与结果汇总
├── hyperparameter_tuning.py     # Optuna超参数优化（可选）
│
├── 模型定义文件（每个模型独立文件）:
│   ├── model_mlp.py             # 多层感知机 (128→128→64→32→2)
│   ├── model_cnn1d.py           # 一维卷积网络 (Conv1d×2 + FC)
│   ├── model_rnn.py             # 双向LSTM循环网络
│   ├── model_cnn_rnn.py         # CNN-RNN混合架构
│   ├── model_mlp_attention.py   # 带注意力机制的MLP
│   ├── model_cnn_residual.py    # 残差连接CNN
│   ├── model_bilstm.py          # 双向长短期记忆网络
│   └── model_gru.py             # 门控循环单元
│
├── model_dl/                    # 训练好的模型权重目录
│   ├── mlp_best.pth            # MLP最佳权重 (~25K参数)
│   ├── cnn1d_best.pth          # CNN1D最佳权重
│   ├── rnn_best.pth            # RNN最佳权重
│   ├── cnn_rnn_best.pth        # CNN-RNN最佳权重
│   ├── mlp_attention_best.pth  # MLP-Attention权重
│   ├── cnn_residual_best.pth   # CNN-Residual权重
│   ├── bilstm_best.pth         # BiLSTM权重
│   └── gru_best.pth            # GRU权重
│
├── results/                     # 实验结果输出目录
    ├── figures/                 # 可视化图表（PNG + PDF格式）
    │   ├── 01_model_comparison.png/pdf         # DL模型性能对比柱状图
    │   ├── 02_training_curves.png/pdf          # 训练曲线汇总图
    │   ├── 02_enhanced_training_curves.png/pdf # 增强版训练曲线对比
    │   ├── 03_confusion_matrices.png/pdf       # 四模型混淆矩阵对比
    │   ├── all_models_comparison.png/pdf       # 所有8个模型综合对比
    │   ├── mlp_training_history.png/pdf        # MLP单独训练曲线
    │   ├── cnn1d_training_history.png/pdf      # CNN1D训练曲线
    │   ├── rnn_training_history.png/pdf        # RNN训练曲线
    │   ├── cnn_rnn_training_history.png/pdf    # CNN-RNN训练曲线
    │   ├── bilstm_training_history.png/pdf     # BiLSTM训练曲线
    │   ├── gru_training_history.png/pdf        # GRU训练曲线
    │   ├── mlp_attention_training_history.png/pdf  # MLP-Attention曲线
    │   └── cnn_residual_training_history.png/pdf   # CNN-Residual曲线
    │
    ├── logs/                     # 训练日志文件
    │   ├── mlp_training.log
    │   ├── cnn1d_training.log
    │   ├── rnn_training.log
    │   ├── cnn_rnn_training.log
    │   ├── ... (共8个日志文件)
    │
    ├── *.csv                     # 训练历史和性能指标
    │   ├── model_comparison.csv           # 最终性能汇总表
    │   ├── all_models_training_history.csv # 所有模型训练历史合并
    │   ├── {model}_training_history.csv   # 各模型单独训练历史
    │   └── training_analysis_summary.csv  # 分析总结
    │
    └── optuna_study.db                   # Optuna优化数据库（如使用）

```

## 🚀 运行指南

### 方式一：一键训练所有模型（推荐）

```bash
cd work_3_dl_classfier
python train_all.py
```

**执行流程**：
1. 加载Work 2提取的128维特征
2. 创建PyTorch DataLoader（batch_size=16）
3. **【基础模型】**依次训练4个核心模型：
   - MLP → CNN1D → RNN → CNN_RNN
4. **【模型变体】**训练4个扩展模型：
   - MLP_Attention → CNN_Residual → BiLSTM → GRU
5. 生成所有模型的训练曲线对比图
6. 保存训练历史到CSV文件
7. 输出性能总结表格

**控制台输出示例**：
```
======================================================================
深度神经网络模型训练
======================================================================

开始训练 MLP 模型
----------------------------------------------------------------------
  设备: cpu (或 cuda)
  训练轮数: 100
  批次大小: 16
  学习率: 0.0002
  
mlp Epochs: 100%|████████████████| 34/34 [00:15<00:00, 2.25it/s]
  Early stopping at epoch 34
  模型已保存: model_dl/mlp_best.pth
  最佳验证准确率: 0.9552

开始训练 CNN1D 模型
----------------------------------------------------------------------
cnn1d Epochs: 100%|████████████████| 28/28 [00:12<00:00, 2.35it/s]
  Early stopping at epoch 21
  ...

======================================================================
所有模型训练完成!
======================================================================

模型性能总结
----------------------------------------------------------------------
【基础模型】
mlp                  0.9552          15.32           25,086
cnn1d                0.9403          12.18           12,544
rnn                  0.6716          45.67           33,280
cnn_rnn              0.9104          89.45           28,672

【模型变体】
mlp_attention        0.9403          22.15           26,112
cnn_residual         0.9254          28.93           18,304
bilstm               0.8955          52.34           66,560
gru                  0.9104          38.76           25,088
```

---

### 方式二：单独训练特定模型

如果只需要训练某个模型或进行调试：

#### 示例：只训练MLP模型
```python
from train_all import *
from model_mlp import create_mlp_model

# 1. 加载数据
X_train, X_test, y_train, y_test = load_features()
train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
input_dim = get_input_dim()

# 2. 创建模型
model = create_mlp_model(input_dim)

# 3. 训练
model, history, val_acc, train_time = train_model(
    model, train_loader, test_loader, 'mlp'
)

print(f"验证准确率: {val_acc:.4f}")
print(f"训练时间: {train_time:.2f}秒")
```

#### 示例：自定义超参数训练
```python
from config import TRAINING_PARAMS

# 修改配置
TRAINING_PARAMS['learning_rate'] = 0.001  # 提高学习率
TRAINING_PARAMS['epochs'] = 50            # 减少训练轮数
TRAINING_PARAMS['batch_size'] = 32        # 增大批次大小

# 然后运行 train_model(...)
```

---

### 方式三：使用已训练模型进行预测

```python
import torch
import numpy as np
from data_loader import load_features
from model_mlp import create_mlp_model

# 1. 加载测试数据
_, X_test, _, y_test = load_features()

# 2. 创建模型并加载权重
input_dim = X_test.shape[1]
model = create_mlp_model(input_dim)

checkpoint = torch.load('model_dl/mlp_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 预测
X_tensor = torch.FloatTensor(X_test)
with torch.no_grad():
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

# 4. 结果分析
accuracy = (predictions == torch.LongTensor(y_test)).float().mean()
print(f"测试准确率: {accuracy:.4f}")

# 单样本预测示例
sample = X_tensor[0:1]
with torch.no_grad():
    pred = model(sample)
    prob = torch.softmax(pred, dim=1)
    
print(f"预测类别: {'猫' if pred.argmax()==0 else '狗'}")
print(f"置信度: {prob.max():.4f}")
print(f"概率分布: 猫={prob[0][0]:.4f}, 狗={prob[0][1]:.4f}")
```

## 📈 训练策略与超参数

### 默认配置（[`config.py`](config.py)）

```python
TRAINING_PARAMS = {
    'epochs': 100,                  # 最大训练轮数
    'batch_size': 16,               # 批次大小
    'learning_rate': 0.0002,        # Adam学习率
    'early_stopping_patience': 10,  # 早停耐心值
    'early_stopping_min_delta': 0.002,  # 最小改善阈值
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
```

## ⚠️ 注意事项

1. **必须先完成Work 2**: 本任务复用Work 2提取的特征（`../work_2_ML_classfier/results/*.npy`），请确保该目录存在且文件完整。

2. **内存需求**: 
   - 数据加载: ~100MB RAM
   - 单模型训练: ~500MB-1GB RAM
   - 全部8个模型: 建议系统内存≥4GB

3. **随机种子**: 已固定为42（见`config.py`），确保可复现性。修改种子会得到略有不同的结果（方差±1-2%）。

4. **Early Stopping**: 大多数模型不会训练满100轮，通常在20-90轮间提前停止。

5. **RNN模型警告**: RNN在该任务上表现异常（67.16%），这是正常的——它揭示了输入表示不匹配的问题，不是代码bug。

## 🐛 常见问题解决

### Q1: 找不到Work 2的特征文件？
```
FileNotFoundError: ../work_2_ML_classfier/results/X_train.npy not found
```
**解决方案**: 先运行Work 2的`run_all.py`生成特征文件。

### Q2: CUDA out of memory?
**解决方案**:
- 减小`batch_size`: 从16改为8或4
- 使用CPU: 设置`DEVICE='cpu'`在`config.py`中

### Q3: 训练不收敛？
**检查项**:
- 学习率是否过大（尝试0.0001或0.00005）
- 是否忘记标准化特征（Work 2已处理）
- 数据标签是否正确（应为0/1整数）

### Q4: 如何添加新模型？
1. 在`model_new.py`中定义模型类
2. 实现`create_new_model(input_dim)`工厂函数
3. 在`train_all.py`的`main()`函数中添加调用
4. 运行即可自动训练并记录结果

### Q5: 如何可视化训练过程？
当前版本在控制台显示进度条(tqdm)。如需TensorBoard：
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
# ... 记录其他指标
```


## 📚 相关文档
- [`config.py`](config.py) - 配置参数说明
- [`data_loader.py`](data_loader.py) - 数据加载流程
- [`train_all.py`](train_all.py) - 主训练脚本详解
- [`evaluate_models.py`](evaluate_models.py) - 评估指标计算方法

各模型实现文件:
- [`model_mlp.py`](model_mlp.py) - MLP架构细节
- [`model_cnn1d.py`](model_cnn1d.py) - CNN1D架构细节
- [`model_rnn.py`](model_rnn.py) - RNN/LSTM架构细节
- [`model_cnn_rnn.py`](model_cnn_rnn.py) - 混合架构设计思路
- 其他变体模型...

---

**最后更新**: 2026-06-22  
