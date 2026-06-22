import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.3, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class CNNWithResidual(nn.Module):
    def __init__(self, input_dim, num_filters=None, kernel_size=None, dropout_rate=None):
        super(CNNWithResidual, self).__init__()
        
        if num_filters is None:
            num_filters = MODEL_PARAMS['cnn1d']['num_filters']
        if kernel_size is None:
            kernel_size = MODEL_PARAMS['cnn1d']['kernel_size']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['cnn1d']['dropout_rate']
        
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(num_filters) - 1):
            self.res_blocks.append(
                ResidualBlock(num_filters[i], num_filters[i+1], kernel_size, dropout_rate)
            )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters[-1], NUM_CLASSES)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_cnn_residual_model(input_dim):
    model = CNNWithResidual(input_dim)
    print(f"CNN-Residual模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  卷积核数量: {MODEL_PARAMS['cnn1d']['num_filters']}")
    print(f"  卷积核大小: {MODEL_PARAMS['cnn1d']['kernel_size']}")
    print(f"  Dropout率: {MODEL_PARAMS['cnn1d']['dropout_rate']}")
    print(f"  残差连接: 启用")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_cnn_residual_model(input_dim)
    print(f"\n模型结构:")
    print(model)
