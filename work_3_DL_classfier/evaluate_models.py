import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import *
from data_loader import load_features, create_data_loaders

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, test_loader, model_name):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm, y_true, y_pred

def plot_model_comparison(all_metrics):
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('accuracy', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    bars = ax.barh(df['model_name'], df['accuracy'], color=colors)
    ax.set_xlabel('准确率', fontsize=12)
    ax.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, value in zip(bars, df['accuracy']):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{value:.4f}', va='center', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    bars = ax.barh(df['model_name'], df['f1'], color=colors)
    ax.set_xlabel('F1分数', fontsize=12)
    ax.set_title('模型F1分数对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, value in zip(bars, df['f1']):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{value:.4f}', va='center', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '01_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比图已保存: {output_path}")

def load_training_history_from_csv(model_name, save_dir=RESULTS_DIR):
    csv_path = os.path.join(save_dir, f'{model_name}_training_history.csv')
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        history = {
            'train_loss': df['train_loss'].tolist(),
            'train_acc': df['train_acc'].tolist(),
            'val_loss': df['val_loss'].tolist(),
            'val_acc': df['val_acc'].tolist()
        }
        
        return history
    except Exception as e:
        print(f"警告: 读取训练历史CSV文件失败 {csv_path}: {e}")
        return None

def analyze_training_history(history, model_name):
    if history is None or not history['train_loss']:
        return None
    
    analysis = {
        'model_name': model_name,
        'total_epochs': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'min_train_loss': min(history['train_loss']),
        'min_val_loss': min(history['val_loss']),
        'max_train_acc': max(history['train_acc']),
        'max_val_acc': max(history['val_acc'])
    }
    
    best_val_acc_idx = history['val_acc'].index(max(history['val_acc']))
    analysis['best_epoch'] = best_val_acc_idx + 1
    analysis['best_val_acc'] = history['val_acc'][best_val_acc_idx]
    analysis['best_val_loss'] = history['val_loss'][best_val_acc_idx]
    analysis['best_train_acc'] = history['train_acc'][best_val_acc_idx]
    analysis['best_train_loss'] = history['train_loss'][best_val_acc_idx]
    
    min_val_loss_idx = history['val_loss'].index(min(history['val_loss']))
    analysis['best_loss_epoch'] = min_val_loss_idx + 1
    
    if len(history['train_loss']) > 1:
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        analysis['loss_reduction_rate'] = (initial_loss - final_loss) / initial_loss * 100
        
        initial_acc = history['train_acc'][0]
        final_acc = history['train_acc'][-1]
        analysis['acc_improvement_rate'] = (final_acc - initial_acc) / initial_acc * 100 if initial_acc > 0 else 0
    else:
        analysis['loss_reduction_rate'] = 0
        analysis['acc_improvement_rate'] = 0
    
    convergence_threshold = 0.001
    for i in range(len(history['val_loss']) - 5):
        window = history['val_loss'][i:i+5]
        if max(window) - min(window) < convergence_threshold:
            analysis['convergence_epoch'] = i + 3
            break
    else:
        analysis['convergence_epoch'] = len(history['train_loss'])
    
    return analysis

def plot_training_curves(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (model_name, result) in enumerate(all_results.items()):
        if idx >= 4:
            break
        
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        csv_history = load_training_history_from_csv(model_name)
        history = csv_history if csv_history is not None else result['history']
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('损失', fontsize=11)
        ax.set_title(f'{model_name.upper()} 训练曲线', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('训练曲线对比', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '02_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线图已保存: {output_path}")

def plot_enhanced_training_curves(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (model_name, result) in enumerate(all_results.items()):
        if idx >= 4:
            break
        
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        csv_history = load_training_history_from_csv(model_name)
        history = csv_history if csv_history is not None else result['history']
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2, alpha=0.8)
        ax.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2, alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.plot(epochs, history['train_acc'], 'b--', label='训练准确率', linewidth=1.5, alpha=0.6)
        ax2.plot(epochs, history['val_acc'], 'r--', label='验证准确率', linewidth=1.5, alpha=0.6)
        ax2.set_ylabel('准确率', fontsize=10, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)
        
        analysis = analyze_training_history(history, model_name)
        if analysis:
            best_epoch = analysis['best_epoch']
            ax.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(best_epoch, ax.get_ylim()[1] * 0.9, f'最佳:{best_epoch}', 
                   fontsize=8, color='green', ha='center')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('损失', fontsize=11)
        ax.set_title(f'{model_name.upper()} 训练曲线', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    
    plt.suptitle('增强训练曲线对比（损失+准确率）', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '02_enhanced_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"增强训练曲线图已保存: {output_path}")

def plot_confusion_matrices(all_cms, model_names):
    n_models = len(model_names)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(zip(model_names, all_cms)):
        if idx >= 4:
            break
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['猫', '狗'], yticklabels=['猫', '狗'])
        ax.set_xlabel('预测标签', fontsize=11)
        ax.set_ylabel('真实标签', fontsize=11)
        ax.set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
    
    for idx in range(n_models, 4):
        axes[idx].axis('off')
    
    plt.suptitle('混淆矩阵对比', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '03_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵图已保存: {output_path}")

def generate_training_analysis_report(all_results):
    analysis_path = os.path.join(RESULTS_DIR, 'training_analysis_report.txt')
    
    all_analyses = []
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("训练历史数据分析报告\n")
        f.write("="*70 + "\n\n")
        
        for model_name, result in all_results.items():
            csv_history = load_training_history_from_csv(model_name)
            history = csv_history if csv_history is not None else result['history']
            
            analysis = analyze_training_history(history, model_name)
            if analysis is None:
                continue
            
            all_analyses.append(analysis)
            
            f.write(f"\n{model_name.upper()} 训练分析:\n")
            f.write("-"*70 + "\n")
            f.write(f"总训练轮数: {analysis['total_epochs']}\n")
            f.write(f"收敛轮数: {analysis['convergence_epoch']}\n\n")
            
            f.write("最佳性能指标:\n")
            f.write(f"  最佳Epoch: {analysis['best_epoch']}\n")
            f.write(f"  最佳验证准确率: {analysis['best_val_acc']:.4f}\n")
            f.write(f"  最佳验证损失: {analysis['best_val_loss']:.4f}\n")
            f.write(f"  对应训练准确率: {analysis['best_train_acc']:.4f}\n")
            f.write(f"  对应训练损失: {analysis['best_train_loss']:.4f}\n\n")
            
            f.write("最终性能指标:\n")
            f.write(f"  最终训练损失: {analysis['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {analysis['final_val_loss']:.4f}\n")
            f.write(f"  最终训练准确率: {analysis['final_train_acc']:.4f}\n")
            f.write(f"  最终验证准确率: {analysis['final_val_acc']:.4f}\n\n")
            
            f.write("训练过程统计:\n")
            f.write(f"  最小训练损失: {analysis['min_train_loss']:.4f}\n")
            f.write(f"  最小验证损失: {analysis['min_val_loss']:.4f}\n")
            f.write(f"  最大训练准确率: {analysis['max_train_acc']:.4f}\n")
            f.write(f"  最大验证准确率: {analysis['max_val_acc']:.4f}\n")
            f.write(f"  损失降低率: {analysis['loss_reduction_rate']:.2f}%\n")
            f.write(f"  准确率提升率: {analysis['acc_improvement_rate']:.2f}%\n\n")
        
        if all_analyses:
            f.write("\n" + "="*70 + "\n")
            f.write("模型训练效率对比\n")
            f.write("="*70 + "\n\n")
            
            sorted_by_epochs = sorted(all_analyses, key=lambda x: x['convergence_epoch'])
            f.write("收敛速度排名（从快到慢）:\n")
            for idx, analysis in enumerate(sorted_by_epochs, 1):
                f.write(f"  {idx}. {analysis['model_name'].upper()}: {analysis['convergence_epoch']} epochs\n")
            
            f.write("\n最佳验证准确率排名:\n")
            sorted_by_acc = sorted(all_analyses, key=lambda x: x['best_val_acc'], reverse=True)
            for idx, analysis in enumerate(sorted_by_acc, 1):
                f.write(f"  {idx}. {analysis['model_name'].upper()}: {analysis['best_val_acc']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("分析报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"训练分析报告已保存: {analysis_path}")
    
    if all_analyses:
        df_analysis = pd.DataFrame(all_analyses)
        csv_analysis_path = os.path.join(RESULTS_DIR, 'training_analysis_summary.csv')
        df_analysis.to_csv(csv_analysis_path, index=False, encoding='utf-8-sig')
        print(f"训练分析摘要已保存: {csv_analysis_path}")
    
    return all_analyses

def generate_report(all_metrics, all_results):
    report_path = os.path.join(WORK_DIR, 'experiment_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("深度神经网络音频分类实验报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、实验环境\n")
        f.write("-"*70 + "\n")
        f.write(f"PyTorch版本: {torch.__version__}\n")
        f.write(f"设备: {DEVICE}\n\n")
        
        f.write("二、模型配置\n")
        f.write("-"*70 + "\n")
        f.write(f"训练轮数: {TRAINING_PARAMS['epochs']}\n")
        f.write(f"批次大小: {TRAINING_PARAMS['batch_size']}\n")
        f.write(f"学习率: {TRAINING_PARAMS['learning_rate']}\n")
        f.write(f"Early Stopping耐心: {TRAINING_PARAMS['early_stopping_patience']}\n\n")
        
        f.write("三、模型性能对比\n")
        f.write("-"*70 + "\n")
        df = pd.DataFrame(all_metrics)
        df = df.sort_values('accuracy', ascending=False)
        
        for idx, row in df.iterrows():
            f.write(f"\n{row['model_name'].upper()}:\n")
            f.write(f"  准确率: {row['accuracy']:.4f}\n")
            f.write(f"  精确率: {row['precision']:.4f}\n")
            f.write(f"  召回率: {row['recall']:.4f}\n")
            f.write(f"  F1分数: {row['f1']:.4f}\n")
            
            if row['model_name'] in all_results:
                res = all_results[row['model_name']]
                f.write(f"  训练时间: {res['train_time']:.2f} 秒\n")
                f.write(f"  参数量: {res['params']:,}\n")
                
                csv_history = load_training_history_from_csv(row['model_name'])
                history = csv_history if csv_history is not None else res['history']
                analysis = analyze_training_history(history, row['model_name'])
                
                if analysis:
                    f.write(f"  最佳Epoch: {analysis['best_epoch']}\n")
                    f.write(f"  收敛轮数: {analysis['convergence_epoch']}\n")
        
        f.write("\n四、最佳模型推荐\n")
        f.write("-"*70 + "\n")
        best_model = df.iloc[0]
        f.write(f"最佳模型: {best_model['model_name'].upper()}\n")
        f.write(f"测试集准确率: {best_model['accuracy']:.4f}\n")
        f.write(f"测试集F1分数: {best_model['f1']:.4f}\n")
        
        f.write("\n五、实验结论\n")
        f.write("-"*70 + "\n")
        f.write("1. 本次实验使用了多种深度神经网络对音频数据进行猫狗分类。\n")
        f.write("2. 所有模型都采用了轻量级设计，适合在CPU上训练。\n")
        f.write("3. 使用了Dropout和Early Stopping防止过拟合。\n")
        f.write("4. 不同架构的模型各有优势，可根据实际需求选择。\n")
        f.write("5. 模型参数量控制在合理范围，训练效率较高。\n")
        f.write("6. 训练历史数据已保存为CSV文件，便于后续分析。\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"实验报告已保存: {report_path}")

def main():
    print("="*70)
    print("模型评估和可视化")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_features()
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    model_files = {
        'mlp': os.path.join(MODEL_DIR, 'mlp_best.pth'),
        'cnn1d': os.path.join(MODEL_DIR, 'cnn1d_best.pth'),
        'rnn': os.path.join(MODEL_DIR, 'rnn_best.pth'),
        'cnn_rnn': os.path.join(MODEL_DIR, 'cnn_rnn_best.pth')
    }
    
    all_metrics = []
    all_cms = []
    all_results = {}
    
    from model_mlp import create_mlp_model
    from model_cnn1d import create_cnn1d_model
    from model_rnn import create_rnn_model
    from model_cnn_rnn import create_cnn_rnn_model
    from data_loader import get_input_dim
    
    input_dim = get_input_dim()
    
    model_creators = {
        'mlp': create_mlp_model,
        'cnn1d': create_cnn1d_model,
        'rnn': create_rnn_model,
        'cnn_rnn': create_cnn_rnn_model
    }
    
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}")
            continue
        
        print(f"\n评估 {model_name.upper()} 模型...")
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        model = model_creators[model_name](input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        
        metrics, cm, y_true, y_pred = evaluate_model(model, test_loader, model_name)
        all_metrics.append(metrics)
        all_cms.append(cm)
        
        all_results[model_name] = {
            'history': checkpoint['history'],
            'train_time': checkpoint['train_time'],
            'params': model.count_parameters()
        }
        
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
    
    print("\n生成可视化图表...")
    plot_model_comparison(all_metrics)
    plot_training_curves(all_results)
    plot_enhanced_training_curves(all_results)
    plot_confusion_matrices(all_cms, list(model_files.keys()))
    
    generate_training_analysis_report(all_results)
    generate_report(all_metrics, all_results)
    
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n性能对比表格已保存: {csv_path}")
    
    print("\n" + "="*70)
    print("评估完成!")
    print("生成的文件:")
    print("  - 模型对比图: figures/01_model_comparison.png")
    print("  - 训练曲线图: figures/02_training_curves.png")
    print("  - 增强训练曲线图: figures/02_enhanced_training_curves.png")
    print("  - 混淆矩阵图: figures/03_confusion_matrices.png")
    print("  - 训练分析报告: results/training_analysis_report.txt")
    print("  - 训练分析摘要: results/training_analysis_summary.csv")
    print("  - 实验报告: experiment_report.txt")
    print("  - 性能对比表格: results/model_comparison.csv")
    print("="*70)

if __name__ == '__main__':
    main()
