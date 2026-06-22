import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=None, num_layers=None, dropout_rate=None):
        super(BiLSTM, self).__init__()
        
        if hidden_size is None:
            hidden_size = MODEL_PARAMS['rnn']['hidden_size']
        if num_layers is None:
            num_layers = MODEL_PARAMS['rnn']['num_layers']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['rnn']['dropout_rate']
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2
        
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(x)
        
        x = lstm_out[:, -1, :]
        x = self.layer_norm(x)
        
        x = self.dropout1(x)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_bilstm_model(input_dim):
    model = BiLSTM(input_dim)
    print(f"BiLSTM模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  LSTM隐藏单元: {MODEL_PARAMS['rnn']['hidden_size']}")
    print(f"  LSTM层数: {MODEL_PARAMS['rnn']['num_layers']}")
    print(f"  双向: True")
    print(f"  Dropout率: {MODEL_PARAMS['rnn']['dropout_rate']}")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_bilstm_model(input_dim)
    print(f"\n模型结构:")
    print(model)
