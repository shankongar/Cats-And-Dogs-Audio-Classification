import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, dropout_rate=None):
        super(MLP, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = MODEL_PARAMS['mlp']['hidden_layers']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['mlp']['dropout_rate']
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, NUM_CLASSES))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_mlp_model(input_dim):
    model = MLP(input_dim)
    print(f"MLP模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {MODEL_PARAMS['mlp']['hidden_layers']}")
    print(f"  Dropout率: {MODEL_PARAMS['mlp']['dropout_rate']}")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_mlp_model(input_dim)
    print(f"\n模型结构:")
    print(model)
