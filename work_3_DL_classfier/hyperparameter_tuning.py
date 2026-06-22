import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from config import *
from data_loader import load_features, create_data_loaders, get_input_dim
from model_mlp import MLP
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    hidden_layer1 = trial.suggest_int('hidden_layer1', 32, 256)
    hidden_layer2 = trial.suggest_int('hidden_layer2', 16, 128)
    hidden_layer3 = trial.suggest_int('hidden_layer3', 8, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    X_train, X_test, y_train, y_test = load_features()
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, batch_size)
    
    input_dim = get_input_dim()
    model = MLP(input_dim, 
                hidden_layers=[hidden_layer1, hidden_layer2, hidden_layer3],
                dropout_rate=dropout_rate)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_acc

def run_hyperparameter_optimization(n_trials=50):
    print("\n" + "="*70)
    print("超参数自动调优 (Optuna)")
    print("="*70)
    print(f"\n优化试验次数: {n_trials}")
    print(f"目标: 最大化验证准确率")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='mlp_hyperparameter_tuning',
        storage=f'sqlite:///{os.path.join(RESULTS_DIR, "optuna_study.db")}',
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "="*70)
    print("优化完成！")
    print("="*70)
    
    print(f"\n最佳验证准确率: {study.best_value:.4f}")
    print(f"\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    fig = plot_optimization_history(study)
    fig.write_html(os.path.join(FIGURES_DIR, 'optuna_optimization_history.html'))
    
    try:
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(FIGURES_DIR, 'optuna_param_importances.html'))
    except:
        print("  警告: 无法生成参数重要性图")
    
    df = study.trials_dataframe()
    csv_path = os.path.join(RESULTS_DIR, 'optuna_trials.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n优化历史已保存: {csv_path}")
    
    best_params_path = os.path.join(RESULTS_DIR, 'best_hyperparameters.txt')
    with open(best_params_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("最佳超参数配置\n")
        f.write("="*60 + "\n\n")
        f.write(f"最佳验证准确率: {study.best_value:.4f}\n\n")
        f.write("最佳超参数:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"最佳超参数已保存: {best_params_path}")
    
    return study

if __name__ == '__main__':
    study = run_hyperparameter_optimization(n_trials=50)
    print("\n超参数调优完成！")
