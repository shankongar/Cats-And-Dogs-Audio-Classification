import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_importance(X_train, y_train, model, model_name, feature_names=None):
    print(f"\n分析 {model_name} 的特征重要性...")
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X_train.shape[1])]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(importances)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'{model_name} - 特征重要性排序', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(FIGURES_DIR, f'{model_name}_feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  特征重要性图已保存: {output_path}")
        
        return importances, indices
    
    return None, None

def plot_feature_importance_comparison(all_models, X_train, feature_names=None):
    print("\n生成特征重要性对比图...")
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X_train.shape[1])]
    
    importance_data = {}
    
    for model_name, model in all_models.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[model_name] = model.feature_importances_
    
    if not importance_data:
        print("  没有可用的特征重要性数据")
        return
    
    df = pd.DataFrame(importance_data, index=feature_names)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.T, cmap='YlOrRd', annot=False, ax=ax, cbar_kws={'label': '重要性'})
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('模型', fontsize=12)
    ax.set_title('各模型特征重要性对比', fontsize=14, fontweight='bold')
    
    output_path = os.path.join(FIGURES_DIR, 'feature_importance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  特征重要性对比图已保存: {output_path}")

def calculate_feature_discrimination(X_train, y_train, feature_names=None):
    print("\n计算特征区分度...")
    
    from scipy import stats
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X_train.shape[1])]
    
    discrimination_scores = []
    
    for i in range(X_train.shape[1]):
        feature_cat = X_train[y_train == 0, i]
        feature_dog = X_train[y_train == 1, i]
        
        t_stat, p_value = stats.ttest_ind(feature_cat, feature_dog)
        
        discrimination_scores.append({
            'feature': feature_names[i],
            't_statistic': abs(t_stat),
            'p_value': p_value,
            'discrimination': abs(t_stat) if p_value < 0.05 else 0
        })
    
    df = pd.DataFrame(discrimination_scores)
    df = df.sort_values('discrimination', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = df.head(20)
    ax.barh(range(len(top_features)), top_features['discrimination'], color='coral')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('区分度（|t统计量|）', fontsize=12)
    ax.set_title('特征区分度排序（Top 20）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, 'feature_discrimination.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  特征区分度图已保存: {output_path}")
    
    csv_path = os.path.join(RESULTS_DIR, 'feature_discrimination.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  特征区分度数据已保存: {csv_path}")
    
    return df

def plot_partial_dependence(model, X_train, model_name, feature_indices=None):
    print(f"\n绘制 {model_name} 的部分依赖图...")
    
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    
    if feature_indices is None:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_indices = np.argsort(importances)[-6:][::-1]
        else:
            feature_indices = list(range(min(6, X_train.shape[1])))
    
    try:
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax = ax.flatten()
        
        display = PartialDependenceDisplay.from_estimator(
            model, X_train, feature_indices, ax=ax, grid_resolution=50
        )
        
        plt.suptitle(f'{model_name} - 部分依赖图', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output_path = os.path.join(FIGURES_DIR, f'{model_name}_pdp.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  部分依赖图已保存: {output_path}")
    except Exception as e:
        print(f"  警告: 无法生成部分依赖图: {e}")

def analyze_shap_values(model, X_train, model_name, sample_size=100):
    print(f"\n分析 {model_name} 的SHAP值...")
    
    try:
        import shap
        
        if X_train.shape[0] > sample_size:
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_sample = X_train[indices]
        else:
            X_sample = X_train
        
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=20)
        plt.title(f'{model_name} - SHAP特征重要性', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(FIGURES_DIR, f'{model_name}_shap_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  SHAP摘要图已保存: {output_path}")
        
        return shap_values
        
    except ImportError:
        print("  警告: shap库未安装，跳过SHAP分析")
        print("  请运行: pip install shap")
        return None
    except Exception as e:
        print(f"  警告: SHAP分析失败: {e}")
        return None

def generate_interpretability_report(X_train, y_train, all_models):
    print("\n" + "="*60)
    print("模型解释性分析")
    print("="*60)
    
    for model_name, model in all_models.items():
        analyze_feature_importance(X_train, y_train, model, model_name)
        plot_partial_dependence(model, X_train, model_name)
        analyze_shap_values(model, X_train, model_name)
    
    plot_feature_importance_comparison(all_models, X_train)
    calculate_feature_discrimination(X_train, y_train)
    
    print("\n" + "="*60)
    print("模型解释性分析完成！")
    print("="*60)

if __name__ == '__main__':
    print("模型解释性分析模块")
    print("使用方法:")
    print("  from model_interpretability import generate_interpretability_report")
    print("  generate_interpretability_report(X_train, y_train, all_models)")
