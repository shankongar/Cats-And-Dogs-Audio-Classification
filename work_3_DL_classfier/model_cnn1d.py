import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_filters=None, kernel_size=None, dropout_rate=None):
        super(CNN1D, self).__init__()
        
        if num_filters is None:
            num_filters = MODEL_PARAMS['cnn1d']['num_filters']
        if kernel_size is None:
            kernel_size = MODEL_PARAMS['cnn1d']['kernel_size']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['cnn1d']['dropout_rate']
        
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        conv_out_size = input_dim // 4
        self.fc1 = nn.Linear(num_filters[1] * conv_out_size, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_cnn1d_model(input_dim):
    model = CNN1D(input_dim)
    print(f"CNN1D模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  卷积滤波器: {MODEL_PARAMS['cnn1d']['num_filters']}")
    print(f"  卷积核大小: {MODEL_PARAMS['cnn1d']['kernel_size']}")
    print(f"  Dropout率: {MODEL_PARAMS['cnn1d']['dropout_rate']}")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_cnn1d_model(input_dim)
    print(f"\n模型结构:")
    print(model)
