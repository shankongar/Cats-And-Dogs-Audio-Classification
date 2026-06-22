import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import FEATURE_DIR, TRAINING_PARAMS, DEVICE

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_features():
    X_train_path = os.path.join(FEATURE_DIR, 'X_train.npy')
    X_test_path = os.path.join(FEATURE_DIR, 'X_test.npy')
    y_train_path = os.path.join(FEATURE_DIR, 'y_train.npy')
    y_test_path = os.path.join(FEATURE_DIR, 'y_test.npy')
    
    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        raise FileNotFoundError(
            "特征文件不存在！请先运行 work_2_ML_classfier/feature_extraction.py 生成特征。"
        )
    
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    print(f"数据加载成功:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
    print(f"  类别分布 - 训练集: 猫={np.sum(y_train==0)}, 狗={np.sum(y_train==1)}")
    print(f"  类别分布 - 测试集: 猫={np.sum(y_test==0)}, 狗={np.sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=None):
    if batch_size is None:
        batch_size = TRAINING_PARAMS['batch_size']
    
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

def get_input_dim():
    X_train_path = os.path.join(FEATURE_DIR, 'X_train.npy')
    if os.path.exists(X_train_path):
        X_train = np.load(X_train_path)
        return X_train.shape[1]
    else:
        raise FileNotFoundError("特征文件不存在！")

if __name__ == '__main__':
    print("测试数据加载器...")
    X_train, X_test, y_train, y_test = load_features()
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    print(f"\nDataLoader创建成功:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    for X_batch, y_batch in train_loader:
        print(f"\n批次示例:")
        print(f"  X shape: {X_batch.shape}")
        print(f"  y shape: {y_batch.shape}")
        break
