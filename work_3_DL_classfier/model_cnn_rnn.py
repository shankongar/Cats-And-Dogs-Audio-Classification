import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class CNN_RNN(nn.Module):
    def __init__(self, input_dim, cnn_filters=None, rnn_hidden=None, dropout_rate=None):
        super(CNN_RNN, self).__init__()
        
        if cnn_filters is None:
            cnn_filters = MODEL_PARAMS['cnn_rnn']['cnn_filters']
        if rnn_hidden is None:
            rnn_hidden = MODEL_PARAMS['cnn_rnn']['rnn_hidden']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['cnn_rnn']['dropout_rate']
        
        self.conv = nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        conv_out_size = input_dim // 2
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(rnn_hidden, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_cnn_rnn_model(input_dim):
    model = CNN_RNN(input_dim)
    print(f"CNN_RNN模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  CNN滤波器: {MODEL_PARAMS['cnn_rnn']['cnn_filters']}")
    print(f"  RNN隐藏单元: {MODEL_PARAMS['cnn_rnn']['rnn_hidden']}")
    print(f"  Dropout率: {MODEL_PARAMS['cnn_rnn']['dropout_rate']}")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_cnn_rnn_model(input_dim)
    print(f"\n模型结构:")
    print(model)
