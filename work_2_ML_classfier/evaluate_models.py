import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import learning_curve
import joblib
from config import *
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_model_comparison(all_metrics):
    print("\n生成模型性能对比图...")
    
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('accuracy', ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    
    ax = axes[2]
    df_auc = df[df['auc'].notna()]
    if len(df_auc) > 0:
        bars = ax.barh(df_auc['model_name'], df_auc['auc'], color=colors[:len(df_auc)])
        ax.set_xlabel('AUC', fontsize=12)
        ax.set_title('模型AUC对比', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        for bar, value in zip(bars, df_auc['auc']):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', va='center', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '01_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_confusion_matrices(all_predictions, y_test, all_metrics):
    print("\n生成混淆矩阵热力图...")
    
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('accuracy', ascending=False)
    top_models = df['model_name'].head(6).tolist()
    
    n_models = len(top_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, model_name in enumerate(top_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        y_pred, _ = all_predictions[model_name]
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['猫', '狗'], yticklabels=['猫', '狗'])
        ax.set_xlabel('预测标签', fontsize=11)
        ax.set_ylabel('真实标签', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('混淆矩阵对比', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '02_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_roc_curves(all_predictions, y_test):
    print("\n生成ROC曲线对比图...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, (y_pred, y_pred_proba) in all_predictions.items():
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)', fontsize=12)
    plt.ylabel('真正率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, '03_roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_feature_importance(all_models, X_train):
    print("\n生成特征重要性分析图...")
    
    tree_models = ['random_forest', 'gradient_boosting', 'adaboost']
    if 'xgboost' in all_models:
        tree_models.append('xgboost')
    if 'lightgbm' in all_models:
        tree_models.append('lightgbm')
    
    available_models = [m for m in tree_models if m in all_models]
    
    if not available_models:
        print("  没有可用的树模型，跳过特征重要性分析")
        return
    
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    n_features = X_train.shape[1]
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    
    for idx, model_name in enumerate(available_models):
        ax = axes[idx]
        model = all_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('重要性', fontsize=11)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('特征重要性对比 (Top 20)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, '04_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def generate_experiment_report(all_metrics, X_train, X_test, y_train, y_test):
    print("\n生成实验报告...")
    
    report_path = os.path.join(WORK_DIR, 'experiment_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("机器学习音频分类实验报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、实验环境\n")
        f.write("-"*70 + "\n")
        import sys
        import sklearn
        import librosa
        f.write(f"Python版本: {sys.version}\n")
        f.write(f"scikit-learn版本: {sklearn.__version__}\n")
        f.write(f"librosa版本: {librosa.__version__}\n")
        f.write(f"numpy版本: {np.__version__}\n")
        f.write(f"pandas版本: {pd.__version__}\n\n")
        
        f.write("二、数据集统计\n")
        f.write("-"*70 + "\n")
        f.write(f"训练集样本数: {len(y_train)}\n")
        f.write(f"测试集样本数: {len(y_test)}\n")
        f.write(f"特征维度: {X_train.shape[1]}\n")
        f.write(f"训练集类别分布: 猫={np.sum(y_train==0)}, 狗={np.sum(y_train==1)}\n")
        f.write(f"测试集类别分布: 猫={np.sum(y_test==0)}, 狗={np.sum(y_test==1)}\n\n")
        
        f.write("三、模型性能对比\n")
        f.write("-"*70 + "\n")
        df = pd.DataFrame(all_metrics)
        df = df.sort_values('accuracy', ascending=False)
        
        for idx, row in df.iterrows():
            f.write(f"\n{row['model_name']}:\n")
            f.write(f"  交叉验证准确率: {row['cv_score']:.4f}\n")
            f.write(f"  测试集准确率: {row['accuracy']:.4f}\n")
            f.write(f"  精确率: {row['precision']:.4f}\n")
            f.write(f"  召回率: {row['recall']:.4f}\n")
            f.write(f"  F1分数: {row['f1']:.4f}\n")
            if row['auc']:
                f.write(f"  AUC: {row['auc']:.4f}\n")
            f.write(f"  训练时间: {row['train_time']:.2f} 秒\n")
            f.write(f"  预测时间: {row['predict_time']:.4f} 秒\n")
            f.write(f"  最佳参数: {row['best_params']}\n")
        
        f.write("\n四、最佳模型推荐\n")
        f.write("-"*70 + "\n")
        best_model = df.iloc[0]
        f.write(f"最佳模型: {best_model['model_name']}\n")
        f.write(f"测试集准确率: {best_model['accuracy']:.4f}\n")
        f.write(f"测试集F1分数: {best_model['f1']:.4f}\n")
        if best_model['auc']:
            f.write(f"测试集AUC: {best_model['auc']:.4f}\n")
        
        f.write("\n五、实验结论\n")
        f.write("-"*70 + "\n")
        f.write("1. 本次实验使用了多种传统机器学习方法对音频数据进行猫狗分类。\n")
        f.write("2. 从音频中提取了丰富的特征，包括MFCC、频谱特征、时域特征等。\n")
        f.write("3. 使用网格搜索对模型超参数进行了优化。\n")
        f.write("4. 集成学习方法（如随机森林、梯度提升）通常表现优于基础模型。\n")
        f.write("5. 最佳模型可以保存并在实际应用中使用。\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"  已保存: {report_path}")

def evaluate_all_models(all_metrics, all_predictions, all_models, X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("开始模型评估和可视化")
    print("="*60)
    
    plot_model_comparison(all_metrics)
    plot_confusion_matrices(all_predictions, y_test, all_metrics)
    plot_roc_curves(all_predictions, y_test)
    plot_feature_importance(all_models, X_train)
    generate_experiment_report(all_metrics, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("模型评估完成！")
    print("="*60)

if __name__ == '__main__':
    from train_models import main as train_main
    all_metrics, all_predictions, all_models, X_test, y_test = train_main()
    
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    
    evaluate_all_models(all_metrics, all_predictions, all_models, X_train, X_test, y_train, y_test)
