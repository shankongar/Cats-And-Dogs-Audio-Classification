import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
from config import *
from data_loader import load_features, create_data_loaders, get_input_dim
from model_mlp import create_mlp_model
from model_cnn1d import create_cnn1d_model
from model_rnn import create_rnn_model
from model_cnn_rnn import create_cnn_rnn_model
from model_mlp_attention import create_mlp_attention_model
from model_cnn_residual import create_cnn_residual_model
from model_bilstm import create_bilstm_model
from model_gru import create_gru_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def setup_logger(model_name, console=False):
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    log_file = os.path.join(LOGS_DIR, f'{model_name}_training.log')
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def plot_training_history(history, model_name, save_dir=FIGURES_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_title(f'{model_name.upper()} - 损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失 (Loss)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax2.set_title(f'{model_name.upper()} - 准确率曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练曲线已保存: {save_path}")
    plt.close()

def plot_all_models_comparison(results, save_dir=FIGURES_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    for idx, name in enumerate(model_names):
        history = results[name]['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], color=colors[idx % len(colors)], 
                label=f'{name.upper()}', linewidth=2)
    ax1.set_title('所有模型 - 训练损失对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('训练损失 (Training Loss)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for idx, name in enumerate(model_names):
        history = results[name]['history']
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], color=colors[idx % len(colors)], 
                label=f'{name.upper()}', linewidth=2)
    ax2.set_title('所有模型 - 验证损失对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('验证损失 (Validation Loss)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for idx, name in enumerate(model_names):
        history = results[name]['history']
        epochs = range(1, len(history['train_acc']) + 1)
        ax3.plot(epochs, history['train_acc'], color=colors[idx % len(colors)], 
                label=f'{name.upper()}', linewidth=2)
    ax3.set_title('所有模型 - 训练准确率对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax3.set_ylabel('训练准确率 (Training Accuracy)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for idx, name in enumerate(model_names):
        history = results[name]['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax4.plot(epochs, history['val_acc'], color=colors[idx % len(colors)], 
                label=f'{name.upper()}', linewidth=2)
    ax4.set_title('所有模型 - 验证准确率对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax4.set_ylabel('验证准确率 (Validation Accuracy)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'all_models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n所有模型对比图已保存: {save_path}")
    plt.close()

def save_training_history_csv(results, save_dir=RESULTS_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for model_name, result in results.items():
        history = result['history']
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc']
        })
        
        csv_path = os.path.join(save_dir, f'{model_name}_training_history.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  {model_name.upper()} 训练历史已保存: {csv_path}")
    
    all_data = []
    for model_name, result in results.items():
        history = result['history']
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        for i, epoch in enumerate(epochs):
            all_data.append({
                'model': model_name,
                'epoch': epoch,
                'train_loss': history['train_loss'][i],
                'train_acc': history['train_acc'][i],
                'val_loss': history['val_loss'][i],
                'val_acc': history['val_acc'][i]
            })
    
    df_all = pd.DataFrame(all_data)
    csv_path_all = os.path.join(save_dir, 'all_models_training_history.csv')
    df_all.to_csv(csv_path_all, index=False, encoding='utf-8-sig')
    print(f"\n所有模型训练历史已保存: {csv_path_all}")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, test_loader, model_name, epochs=None):
    if epochs is None:
        epochs = TRAINING_PARAMS['epochs']
    
    logger = setup_logger(model_name)
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])
    
    early_stopping = EarlyStopping(
        patience=TRAINING_PARAMS['early_stopping_patience'],
        min_delta=TRAINING_PARAMS['early_stopping_min_delta']
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    logger.info("=" * 80)
    logger.info(f"开始训练模型: {model_name}")
    logger.info("=" * 80)
    logger.info(f"训练配置:")
    logger.info(f"  - 设备: {DEVICE}")
    logger.info(f"  - 训练轮数: {epochs}")
    logger.info(f"  - 批次大小: {TRAINING_PARAMS['batch_size']}")
    logger.info(f"  - 学习率: {TRAINING_PARAMS['learning_rate']}")
    logger.info(f"  - 早停耐心值: {TRAINING_PARAMS['early_stopping_patience']}")
    logger.info(f"  - 早停最小差值: {TRAINING_PARAMS['early_stopping_min_delta']}")
    logger.info("=" * 80)
    
    tqdm.write(f"\n开始训练 {model_name}...")
    tqdm.write(f"  设备: {DEVICE}")
    tqdm.write(f"  训练轮数: {epochs}")
    tqdm.write(f"  批次大小: {TRAINING_PARAMS['batch_size']}")
    tqdm.write(f"  学习率: {TRAINING_PARAMS['learning_rate']}")
    
    start_time = time.time()
    
    epoch_pbar = tqdm(range(epochs), desc=f'{model_name} Epochs')
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'  Training', leave=False)
        for X_batch, y_batch in train_pbar:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
            batch_loss = loss.item()
            batch_acc = (predicted == y_batch).sum().item() / y_batch.size(0)
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'acc': f'{batch_acc:.4f}'})
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'  Validating', leave=False)
            for X_batch, y_batch in val_pbar:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                batch_loss = loss.item()
                batch_acc = (predicted == y_batch).sum().item() / y_batch.size(0)
                val_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'acc': f'{batch_acc:.4f}'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                   f"LR: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            tqdm.write(f"  Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_val_acc': best_val_acc,
        'train_time': train_time
    }, model_path)
    
    logger.info("=" * 80)
    logger.info(f"训练完成!")
    logger.info(f"  - 总用时: {train_time:.2f} 秒")
    logger.info(f"  - 最佳验证准确率: {best_val_acc:.4f}")
    logger.info(f"  - 模型已保存: {model_path}")
    logger.info(f"  - 训练轮数: {len(history['train_loss'])}")
    logger.info("=" * 80)
    
    tqdm.write(f"  模型已保存: {model_path}")
    tqdm.write(f"  训练完成! 用时: {train_time:.2f}秒")
    tqdm.write(f"  最佳验证准确率: {best_val_acc:.4f}\n")
    
    plot_training_history(history, model_name)
    
    return model, history, best_val_acc, train_time

def main():
    print("="*70)
    print("深度神经网络模型训练")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_features()
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    input_dim = get_input_dim()
    
    results = {}
    
    print("\n" + "="*70)
    print("【基础模型】")
    print("="*70)
    
    print("\n训练 MLP 模型")
    print("-"*70)
    model_mlp = create_mlp_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_mlp, train_loader, test_loader, 'mlp')
    results['mlp'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 CNN1D 模型")
    print("-"*70)
    model_cnn1d = create_cnn1d_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_cnn1d, train_loader, test_loader, 'cnn1d')
    results['cnn1d'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 RNN 模型")
    print("-"*70)
    model_rnn = create_rnn_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_rnn, train_loader, test_loader, 'rnn')
    results['rnn'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 CNN_RNN 模型")
    print("-"*70)
    model_cnn_rnn = create_cnn_rnn_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_cnn_rnn, train_loader, test_loader, 'cnn_rnn')
    results['cnn_rnn'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n" + "="*70)
    print("【模型变体】")
    print("="*70)
    
    print("\n训练 MLP-Attention 模型")
    print("-"*70)
    model_mlp_att = create_mlp_attention_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_mlp_att, train_loader, test_loader, 'mlp_attention')
    results['mlp_attention'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 CNN-Residual 模型")
    print("-"*70)
    model_cnn_res = create_cnn_residual_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_cnn_res, train_loader, test_loader, 'cnn_residual')
    results['cnn_residual'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 BiLSTM 模型")
    print("-"*70)
    model_bilstm = create_bilstm_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_bilstm, train_loader, test_loader, 'bilstm')
    results['bilstm'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n训练 GRU 模型")
    print("-"*70)
    model_gru = create_gru_model(input_dim)
    model, history, val_acc, train_time = train_model(
        model_gru, train_loader, test_loader, 'gru')
    results['gru'] = {
        'model': model,
        'history': history,
        'val_acc': val_acc,
        'train_time': train_time,
        'params': model.count_parameters()
    }
    
    print("\n" + "="*70)
    print("所有模型训练完成!")
    print("="*70)
    
    print("\n生成训练曲线对比图...")
    plot_all_models_comparison(results)
    
    print("\n保存训练历史数据到CSV...")
    save_training_history_csv(results)
    
    print("\n" + "="*70)
    print("模型性能总结")
    print("="*70)
    print(f"\n{'模型':<20} {'验证准确率':<15} {'训练时间(秒)':<15} {'参数量':<15}")
    print("-" * 65)
    
    basic_models = ['mlp', 'cnn1d', 'rnn', 'cnn_rnn']
    variant_models = ['mlp_attention', 'cnn_residual', 'bilstm', 'gru']
    
    print("\n【基础模型】")
    for name in basic_models:
        if name in results:
            res = results[name]
            print(f"{name:<20} {res['val_acc']:<15.4f} {res['train_time']:<15.2f} {res['params']:<15,}")
    
    print("\n【模型变体】")
    for name in variant_models:
        if name in results:
            res = results[name]
            print(f"{name:<20} {res['val_acc']:<15.4f} {res['train_time']:<15.2f} {res['params']:<15,}")
    
    return results

if __name__ == '__main__':
    results = main()
