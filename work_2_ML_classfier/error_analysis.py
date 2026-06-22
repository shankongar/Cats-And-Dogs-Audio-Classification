import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR, CLASS_LABELS
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_errors(y_true, y_pred, model_name, X_test=None):
    print(f"\n分析 {model_name} 的错误样本...")
    
    error_indices = np.where(y_true != y_pred)[0]
    correct_indices = np.where(y_true == y_pred)[0]
    
    error_analysis = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_true),
        'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
        'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
    }
    
    print(f"  总错误数: {error_analysis['total_errors']}")
    print(f"  错误率: {error_analysis['error_rate']:.4f}")
    print(f"  假正例（猫预测为狗）: {error_analysis['false_positives']}")
    print(f"  假负例（狗预测为猫）: {error_analysis['false_negatives']}")
    
    return error_analysis, error_indices

def plot_error_distribution(all_predictions, y_test):
    print("\n绘制错误分布对比图...")
    
    error_data = []
    
    for model_name, (y_pred, _) in all_predictions.items():
        error_indices = np.where(y_test != y_pred)[0]
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        error_data.append({
            'model': model_name,
            'total_errors': len(error_indices),
            'false_positives': fp,
            'false_negatives': fn
        })
    
    df = pd.DataFrame(error_data)
    df = df.sort_values('total_errors', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.barh(x - width/2, df['false_positives'], width, 
           label='假正例（猫→狗）', color='#FF6B6B', alpha=0.8)
    ax.barh(x + width/2, df['false_negatives'], width, 
           label='假负例（狗→猫）', color='#4ECDC4', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df['model'])
    ax.set_xlabel('错误数量', fontsize=12)
    ax.set_title('模型错误类型分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, 'error_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def identify_hard_samples(all_predictions, y_test):
    print("\n识别困难样本...")
    
    n_samples = len(y_test)
    error_count = np.zeros(n_samples)
    
    for model_name, (y_pred, _) in all_predictions.items():
        error_count += (y_test != y_pred).astype(int)
    
    hard_samples = np.where(error_count >= len(all_predictions) * 0.5)[0]
    easy_samples = np.where(error_count == 0)[0]
    
    print(f"  困难样本数（多数模型预测错误）: {len(hard_samples)}")
    print(f"  简单样本数（所有模型预测正确）: {len(easy_samples)}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(error_count, bins=range(int(error_count.max()) + 2), 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('预测错误模型数', fontsize=12)
    ax.set_ylabel('样本数', fontsize=12)
    ax.set_title('样本难度分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, 'sample_difficulty.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    return hard_samples, easy_samples, error_count

def generate_error_report(all_predictions, y_test):
    print("\n生成错误分析报告...")
    
    report_path = os.path.join(RESULTS_DIR, 'error_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("错误分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、各模型错误统计\n")
        f.write("-"*70 + "\n")
        
        for model_name, (y_pred, _) in all_predictions.items():
            error_indices = np.where(y_test != y_pred)[0]
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            f.write(f"\n{model_name}:\n")
            f.write(f"  总错误数: {len(error_indices)}\n")
            f.write(f"  错误率: {len(error_indices)/len(y_test):.4f}\n")
            f.write(f"  假正例（猫→狗）: {fp}\n")
            f.write(f"  假负例（狗→猫）: {fn}\n")
        
        hard_samples, easy_samples, error_count = identify_hard_samples(all_predictions, y_test)
        
        f.write("\n\n二、样本难度分析\n")
        f.write("-"*70 + "\n")
        f.write(f"困难样本数（多数模型预测错误）: {len(hard_samples)}\n")
        f.write(f"简单样本数（所有模型预测正确）: {len(easy_samples)}\n")
        
        if len(hard_samples) > 0:
            f.write(f"\n困难样本索引: {hard_samples[:20].tolist()}\n")
        
        f.write("\n\n三、改进建议\n")
        f.write("-"*70 + "\n")
        f.write("1. 分析困难样本的特征，寻找共性\n")
        f.write("2. 针对错误类型调整模型权重\n")
        f.write("3. 考虑增加困难样本的数据增强\n")
        f.write("4. 集成多个模型以减少错误\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"  已保存: {report_path}")

def main():
    print("\n" + "="*70)
    print("错误分析")
    print("="*70)
    
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    all_predictions = {}
    
    model_files = {
        'logistic_regression': os.path.join(MODEL_DIR, 'logistic_regression_best.pkl'),
        'svm': os.path.join(MODEL_DIR, 'svm_best.pkl'),
        'random_forest': os.path.join(MODEL_DIR, 'random_forest_best.pkl')
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            all_predictions[model_name] = (y_pred, y_pred_proba)
            
            analyze_errors(y_test, y_pred, model_name)
    
    if all_predictions:
        plot_error_distribution(all_predictions, y_test)
        generate_error_report(all_predictions, y_test)
    
    print("\n" + "="*70)
    print("错误分析完成！")
    print("="*70)

if __name__ == '__main__':
    main()
