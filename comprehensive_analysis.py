import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'experiment_summary')

def load_all_results():
    results = []
    
    ml_csv = os.path.join(BASE_DIR, 'work_2_ML_classfier', 'results', 'model_comparison.csv')
    if os.path.exists(ml_csv):
        df_ml = pd.read_csv(ml_csv)
        df_ml['category'] = '传统机器学习'
        results.append(df_ml)
        print(f"加载ML结果: {len(df_ml)} 个模型")
    
    dl_csv = os.path.join(BASE_DIR, 'work_3_DL_classfier', 'results', 'model_comparison.csv')
    if os.path.exists(dl_csv):
        df_dl = pd.read_csv(dl_csv)
        df_dl['category'] = '深度学习'
        results.append(df_dl)
        print(f"加载DL结果: {len(df_dl)} 个模型")
    
    if results:
        all_results = pd.concat(results, ignore_index=True)
        return all_results
    else:
        print("警告: 未找到任何结果文件")
        return None

def plot_comprehensive_comparison(df):
    print("\n生成综合对比图...")
    
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    colors = ['#FF6B6B' if cat == '传统机器学习' else '#4ECDC4' for cat in df_sorted['category']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['model_name'])
    ax.set_xlabel('准确率', fontsize=12)
    ax.set_title('所有模型准确率对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}', va='center', fontsize=9)
    
    ax = axes[0, 1]
    bars = ax.barh(range(len(df_sorted)), df_sorted['f1'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['model_name'])
    ax.set_xlabel('F1分数', fontsize=12)
    ax.set_title('所有模型F1分数对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    category_stats = df.groupby('category').agg({
        'accuracy': ['mean', 'max', 'std'],
        'f1': ['mean', 'max', 'std']
    }).round(4)
    
    categories = df['category'].unique()
    x = np.arange(len(categories))
    width = 0.35
    
    mean_acc = df.groupby('category')['accuracy'].mean()
    max_acc = df.groupby('category')['accuracy'].max()
    
    ax.bar(x - width/2, mean_acc, width, label='平均准确率', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, max_acc, width, label='最高准确率', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('方法类别', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('不同方法类别的性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if 'train_time' in df.columns:
        df_time = df[df['train_time'].notna()].copy()
        df_time = df_time.sort_values('train_time', ascending=True)
        
        ax.barh(range(len(df_time)), df_time['train_time'], 
               color=['#FF6B6B' if cat == '传统机器学习' else '#4ECDC4' 
                     for cat in df_time['category']])
        ax.set_yticks(range(len(df_time)))
        ax.set_yticklabels(df_time['model_name'])
        ax.set_xlabel('训练时间 (秒)', fontsize=12)
        ax.set_title('模型训练时间对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('跨任务模型综合对比', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def generate_summary_report(df):
    print("\n生成综合报告...")
    
    report_path = os.path.join(OUTPUT_DIR, 'experiment_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("音频分类实验综合报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("一、实验概述\n")
        f.write("-"*80 + "\n")
        f.write(f"总模型数: {len(df)}\n")
        f.write(f"方法类别: {', '.join(df['category'].unique())}\n\n")
        
        f.write("二、性能排名\n")
        f.write("-"*80 + "\n")
        df_sorted = df.sort_values('accuracy', ascending=False)
        
        f.write("\nTop 5 模型:\n")
        for i, row in df_sorted.head(5).iterrows():
            f.write(f"  {row['model_name']}: 准确率={row['accuracy']:.4f}, "
                   f"F1={row['f1']:.4f}, 类别={row['category']}\n")
        
        f.write("\n三、类别对比\n")
        f.write("-"*80 + "\n")
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            f.write(f"\n{category}:\n")
            f.write(f"  模型数量: {len(cat_df)}\n")
            f.write(f"  平均准确率: {cat_df['accuracy'].mean():.4f}\n")
            f.write(f"  最高准确率: {cat_df['accuracy'].max():.4f}\n")
            f.write(f"  准确率标准差: {cat_df['accuracy'].std():.4f}\n")
        
        f.write("\n四、最佳模型推荐\n")
        f.write("-"*80 + "\n")
        best_model = df_sorted.iloc[0]
        f.write(f"最佳模型: {best_model['model_name']}\n")
        f.write(f"类别: {best_model['category']}\n")
        f.write(f"准确率: {best_model['accuracy']:.4f}\n")
        f.write(f"精确率: {best_model['precision']:.4f}\n")
        f.write(f"召回率: {best_model['recall']:.4f}\n")
        f.write(f"F1分数: {best_model['f1']:.4f}\n")
        if 'auc' in best_model and pd.notna(best_model['auc']):
            f.write(f"AUC: {best_model['auc']:.4f}\n")
        
        f.write("\n五、实验结论\n")
        f.write("-"*80 + "\n")
        
        ml_best = df[df['category'] == '传统机器学习']['accuracy'].max()
        dl_best = df[df['category'] == '深度学习']['accuracy'].max()
        
        if ml_best > dl_best:
            f.write(f"1. 传统机器学习方法表现最优，最高准确率 {ml_best:.4f}\n")
        else:
            f.write(f"1. 深度学习方法表现最优，最高准确率 {dl_best:.4f}\n")
        
        f.write("2. 不同方法各有优势，应根据实际需求选择:\n")
        f.write("   - 传统ML: 训练快、可解释性强、数据需求少\n")
        f.write("   - 深度学习: 表达能力强、自动特征学习\n")
        f.write("3. 集成方法通常优于单一模型\n")
        f.write("4. 特征工程对性能影响显著\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"  已保存: {report_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, 'all_models_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  已保存: {csv_path}")

def main():
    print("\n" + "="*70)
    print("跨任务综合对比分析")
    print("="*70)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    df = load_all_results()
    
    if df is not None:
        plot_comprehensive_comparison(df)
        generate_summary_report(df)
        
        print("\n" + "="*70)
        print("综合对比分析完成！")
        print("="*70)
        print(f"\n结果保存在: {OUTPUT_DIR}")
    else:
        print("\n错误: 无法加载任何结果数据")

if __name__ == '__main__':
    main()
