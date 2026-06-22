import torch
import torch.nn as nn
from config import MODEL_PARAMS, NUM_CLASSES

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_weights, V)
        return output

class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, dropout_rate=None, use_attention=True):
        super(MLPWithAttention, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = MODEL_PARAMS['mlp']['hidden_layers']
        if dropout_rate is None:
            dropout_rate = MODEL_PARAMS['mlp']['dropout_rate']
        
        self.use_attention = use_attention
        self.hidden_layers = hidden_layers
        
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.input_bn = nn.BatchNorm1d(hidden_layers[0])
        self.input_dropout = nn.Dropout(dropout_rate)
        
        self.hidden_layers_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers_list.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_layers[i+1]))
            if use_attention:
                self.attentions.append(SelfAttention(hidden_layers[i+1]))
        
        self.output_layer = nn.Linear(hidden_layers[-1], NUM_CLASSES)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.input_layer(x)
        if x.size(0) > 1:
            x = self.input_bn(x)
        x = self.relu(x)
        x = self.input_dropout(x)
        
        for i, (linear, bn) in enumerate(zip(self.hidden_layers_list, self.batch_norms)):
            x = linear(x)
            if x.size(0) > 1:
                x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            if self.use_attention and i < len(self.attentions):
                x_unsqueeze = x.unsqueeze(1)
                x_att = self.attentions[i](x_unsqueeze)
                x = x + x_att.squeeze(1)
        
        x = self.output_layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_mlp_attention_model(input_dim):
    model = MLPWithAttention(input_dim)
    print(f"MLP-Attention模型创建成功:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {MODEL_PARAMS['mlp']['hidden_layers']}")
    print(f"  Dropout率: {MODEL_PARAMS['mlp']['dropout_rate']}")
    print(f"  注意力机制: 启用")
    print(f"  可训练参数: {model.count_parameters():,}")
    return model

if __name__ == '__main__':
    from data_loader import get_input_dim
    input_dim = get_input_dim()
    model = create_mlp_attention_model(input_dim)
    print(f"\n模型结构:")
    print(model)
