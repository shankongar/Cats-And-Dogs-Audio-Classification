import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FEATURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'work_2_ML_classfier', 'results')
OUTPUT_DIR = os.path.dirname(__file__)

def load_features():
    X_train_path = os.path.join(FEATURE_DIR, 'X_train.npy')
    X_test_path = os.path.join(FEATURE_DIR, 'X_test.npy')
    y_train_path = os.path.join(FEATURE_DIR, 'y_train.npy')
    y_test_path = os.path.join(FEATURE_DIR, 'y_test.npy')
    
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    return X_train, X_test, y_train, y_test

def plot_feature_correlation(X_train, feature_names=None):
    print("\n绘制特征相关性热力图...")
    
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(X_train.shape[1])]
    
    corr_matrix = np.corrcoef(X_train.T)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                xticklabels=feature_names, yticklabels=feature_names,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('特征相关性矩阵', fontsize=16, fontweight='bold')
    
    output_path = os.path.join(OUTPUT_DIR, '06_feature_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    return corr_matrix

def plot_pca_2d(X_train, y_train):
    print("\n绘制PCA 2D可视化...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cat_mask = y_train == 0
    dog_mask = y_train == 1
    
    ax.scatter(X_pca[cat_mask, 0], X_pca[cat_mask, 1], 
               c='#FF6B6B', label='猫', alpha=0.6, s=50)
    ax.scatter(X_pca[dog_mask, 0], X_pca[dog_mask, 1], 
               c='#4ECDC4', label='狗', alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title('PCA降维可视化 (2D)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(OUTPUT_DIR, '07_pca_2d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    print(f"  总解释方差: {sum(pca.explained_variance_ratio_):.2%}")
    
    return pca

def plot_pca_3d(X_train, y_train):
    print("\n绘制PCA 3D可视化...")
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_train)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    cat_mask = y_train == 0
    dog_mask = y_train == 1
    
    ax.scatter(X_pca[cat_mask, 0], X_pca[cat_mask, 1], X_pca[cat_mask, 2],
               c='#FF6B6B', label='猫', alpha=0.6, s=50)
    ax.scatter(X_pca[dog_mask, 0], X_pca[dog_mask, 1], X_pca[dog_mask, 2],
               c='#4ECDC4', label='狗', alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
    ax.set_title('PCA降维可视化 (3D)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    
    output_path = os.path.join(OUTPUT_DIR, '08_pca_3d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_tsne(X_train, y_train, perplexity=30):
    print(f"\n绘制t-SNE可视化 (perplexity={perplexity})...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_train)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cat_mask = y_train == 0
    dog_mask = y_train == 1
    
    ax.scatter(X_tsne[cat_mask, 0], X_tsne[cat_mask, 1], 
               c='#FF6B6B', label='猫', alpha=0.6, s=50)
    ax.scatter(X_tsne[dog_mask, 0], X_tsne[dog_mask, 1], 
               c='#4ECDC4', label='狗', alpha=0.6, s=50)
    
    ax.set_xlabel('t-SNE维度1', fontsize=12)
    ax.set_ylabel('t-SNE维度2', fontsize=12)
    ax.set_title(f't-SNE可视化 (perplexity={perplexity})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(OUTPUT_DIR, '09_tsne.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def analyze_feature_discrimination(X_train, y_train, feature_names=None):
    print("\n分析特征区分度...")
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X_train.shape[1])]
    
    discrimination_scores = []
    
    for i in range(X_train.shape[1]):
        feature_cat = X_train[y_train == 0, i]
        feature_dog = X_train[y_train == 1, i]
        
        t_stat, p_value = stats.ttest_ind(feature_cat, feature_dog)
        
        effect_size = abs(t_stat) / np.sqrt(len(feature_cat) + len(feature_dog))
        
        discrimination_scores.append({
            'feature': feature_names[i],
            'feature_idx': i,
            't_statistic': abs(t_stat),
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        })
    
    df = pd.DataFrame(discrimination_scores)
    df = df.sort_values('t_statistic', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = df.head(20)
    colors = ['#FF6B6B' if sig else '#95a5a6' for sig in top_features['significant']]
    ax.barh(range(len(top_features)), top_features['t_statistic'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('|t统计量|', fontsize=12)
    ax.set_title('特征区分度排序 (Top 20, 红色=显著)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(OUTPUT_DIR, '10_feature_discrimination.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, 'feature_discrimination.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  数据已保存: {csv_path}")
    
    return df

def main():
    print("\n" + "="*70)
    print("深度特征分析")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_features()
    
    print(f"\n数据集信息:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    plot_feature_correlation(X_train)
    plot_pca_2d(X_train, y_train)
    plot_pca_3d(X_train, y_train)
    plot_tsne(X_train, y_train, perplexity=30)
    analyze_feature_discrimination(X_train, y_train)
    
    print("\n" + "="*70)
    print("深度特征分析完成！")
    print("="*70)

if __name__ == '__main__':
    main()
