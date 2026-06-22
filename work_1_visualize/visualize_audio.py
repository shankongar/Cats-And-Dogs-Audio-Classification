import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'cats_dogs')
OUTPUT_DIR = os.path.dirname(__file__)
SR = 16000

def get_audio_files():
    audio_files = {
        'train_cat': glob.glob(os.path.join(DATA_DIR, 'train', 'cat', '*.wav')),
        'train_dog': glob.glob(os.path.join(DATA_DIR, 'train', 'dog', '*.wav')),
        'test_cat': glob.glob(os.path.join(DATA_DIR, 'test', 'cats', '*.wav')),
        'test_dog': glob.glob(os.path.join(DATA_DIR, 'test', 'test', '*.wav'))
    }
    
    if not audio_files['train_cat']:
        audio_files['train_cat'] = glob.glob(os.path.join(DATA_DIR, 'cat_*.wav'))[:100]
    if not audio_files['train_dog']:
        audio_files['train_dog'] = glob.glob(os.path.join(DATA_DIR, 'dog_barking_*.wav'))[:100]
    if not audio_files['test_cat']:
        audio_files['test_cat'] = glob.glob(os.path.join(DATA_DIR, 'cat_*.wav'))[100:130]
    if not audio_files['test_dog']:
        audio_files['test_dog'] = glob.glob(os.path.join(DATA_DIR, 'dog_barking_*.wav'))[100:130]
    
    return audio_files

def get_audio_duration(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        return len(y) / sr
    except:
        return 0

def plot_data_statistics(audio_files):
    print("正在生成数据统计可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    counts = {
        '训练集-猫': len(audio_files['train_cat']),
        '训练集-狗': len(audio_files['train_dog']),
        '测试集-猫': len(audio_files['test_cat']),
        '测试集-狗': len(audio_files['test_dog'])
    }
    
    ax1 = axes[0, 0]
    categories = list(counts.keys())
    values = list(counts.values())
    bars = ax1.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_ylabel('文件数量', fontsize=12)
    ax1.set_title('各类别音频文件数量统计', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[0, 1]
    train_cat_count = len(audio_files['train_cat'])
    train_dog_count = len(audio_files['train_dog'])
    test_cat_count = len(audio_files['test_cat'])
    test_dog_count = len(audio_files['test_dog'])
    
    train_total = train_cat_count + train_dog_count
    test_total = test_cat_count + test_dog_count
    
    sizes = [train_total, test_total]
    labels = [f'训练集\n({train_total})', f'测试集\n({test_total})']
    colors = ['#95E1D3', '#F38181']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('训练集与测试集比例', fontsize=14, fontweight='bold')
    
    ax3 = axes[1, 0]
    print("  计算音频时长分布...")
    durations = {'猫': [], '狗': []}
    
    sample_files = {
        '猫': (audio_files['train_cat'][:20] + audio_files['test_cat'][:10]),
        '狗': (audio_files['train_dog'][:20] + audio_files['test_dog'][:10])
    }
    
    for label, files in sample_files.items():
        for file in files:
            duration = get_audio_duration(file)
            if duration > 0:
                durations[label].append(duration)
    
    if durations['猫'] and durations['狗']:
        from scipy import stats as sp_stats
        
        cat_durations = np.array(durations['猫'])
        dog_durations = np.array(durations['狗'])
        
        x_min = min(cat_durations.min(), dog_durations.min())
        x_max = max(cat_durations.max(), dog_durations.max())
        x_range = np.linspace(x_min, x_max, 200)
        
        kde_cat = sp_stats.gaussian_kde(cat_durations)
        kde_dog = sp_stats.gaussian_kde(dog_durations)
        
        ax3.plot(x_range, kde_cat(x_range), label='猫', color='#FF6B6B', linewidth=2.5)
        ax3.fill_between(x_range, kde_cat(x_range), alpha=0.3, color='#FF6B6B')
        ax3.plot(x_range, kde_dog(x_range), label='狗', color='#4ECDC4', linewidth=2.5)
        ax3.fill_between(x_range, kde_dog(x_range), alpha=0.3, color='#4ECDC4')
        ax3.set_xlabel('时长 (秒)', fontsize=12)
        ax3.set_ylabel('概率密度', fontsize=12)
        ax3.set_title('音频时长分布对比 (KDE)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    cat_total = train_cat_count + test_cat_count
    dog_total = train_dog_count + test_dog_count
    sizes = [cat_total, dog_total]
    labels = [f'猫\n({cat_total})', f'狗\n({dog_total})']
    colors = ['#FF6B6B', '#4ECDC4']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11}, explode=(0.05, 0.05))
    ax4.set_title('猫与狗音频总数量比例', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '01_data_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    return counts, durations

def plot_waveform_samples(audio_files):
    print("正在生成音频波形可视化...")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    cat_files = (audio_files['train_cat'][:2] + audio_files['test_cat'][:2])
    dog_files = (audio_files['train_dog'][:2] + audio_files['test_dog'][:2])
    
    for i, (cat_file, dog_file) in enumerate(zip(cat_files, dog_files)):
        y_cat, _ = librosa.load(cat_file, sr=SR)
        y_dog, _ = librosa.load(dog_file, sr=SR)
        
        ax_cat = axes[i, 0]
        time_cat = np.linspace(0, len(y_cat)/SR, len(y_cat))
        ax_cat.plot(time_cat, y_cat, color='#FF6B6B', linewidth=0.5)
        ax_cat.set_xlabel('时间 (秒)', fontsize=10)
        ax_cat.set_ylabel('振幅', fontsize=10)
        ax_cat.set_title(f'猫音频样本 {i+1}', fontsize=12, fontweight='bold')
        ax_cat.grid(True, alpha=0.3)
        
        ax_dog = axes[i, 1]
        time_dog = np.linspace(0, len(y_dog)/SR, len(y_dog))
        ax_dog.plot(time_dog, y_dog, color='#4ECDC4', linewidth=0.5)
        ax_dog.set_xlabel('时间 (秒)', fontsize=10)
        ax_dog.set_ylabel('振幅', fontsize=10)
        ax_dog.set_title(f'狗音频样本 {i+1}', fontsize=12, fontweight='bold')
        ax_dog.grid(True, alpha=0.3)
    
    plt.suptitle('音频波形对比 - 猫 vs 狗', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '02_waveform_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_spectrogram_samples(audio_files):
    print("正在生成频谱分析可视化...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    cat_files = audio_files['train_cat'][:3]
    dog_files = audio_files['train_dog'][:3]
    
    for i, (cat_file, dog_file) in enumerate(zip(cat_files, dog_files)):
        y_cat, _ = librosa.load(cat_file, sr=SR)
        y_dog, _ = librosa.load(dog_file, sr=SR)
        
        D_cat = np.abs(librosa.stft(y_cat))
        D_dog = np.abs(librosa.stft(y_dog))
        
        ax = axes[i, 0]
        librosa.display.specshow(librosa.amplitude_to_db(D_cat, ref=np.max),
                                 y_axis='log', x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'猫 - 频谱图 {i+1}', fontsize=11, fontweight='bold')
        
        ax = axes[i, 1]
        S_cat = librosa.feature.melspectrogram(y=y_cat, sr=SR)
        S_dog = librosa.feature.melspectrogram(y=y_dog, sr=SR)
        librosa.display.specshow(librosa.power_to_db(S_cat, ref=np.max),
                                 y_axis='mel', x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'猫 - 梅尔频谱 {i+1}', fontsize=11, fontweight='bold')
        
        ax = axes[i, 2]
        librosa.display.specshow(librosa.amplitude_to_db(D_dog, ref=np.max),
                                 y_axis='log', x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'狗 - 频谱图 {i+1}', fontsize=11, fontweight='bold')
        
        ax = axes[i, 3]
        librosa.display.specshow(librosa.power_to_db(S_dog, ref=np.max),
                                 y_axis='mel', x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'狗 - 梅尔频谱 {i+1}', fontsize=11, fontweight='bold')
    
    plt.suptitle('频谱分析对比 - 猫 vs 狗', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '03_spectrogram_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def plot_mfcc_samples(audio_files):
    print("正在生成MFCC特征可视化...")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    
    cat_files = (audio_files['train_cat'][:2] + audio_files['test_cat'][:2])
    dog_files = (audio_files['train_dog'][:2] + audio_files['test_dog'][:2])
    
    for i, (cat_file, dog_file) in enumerate(zip(cat_files, dog_files)):
        y_cat, _ = librosa.load(cat_file, sr=SR)
        y_dog, _ = librosa.load(dog_file, sr=SR)
        
        mfcc_cat = librosa.feature.mfcc(y=y_cat, sr=SR, n_mfcc=13)
        mfcc_dog = librosa.feature.mfcc(y=y_dog, sr=SR, n_mfcc=13)
        
        ax = axes[i, 0]
        librosa.display.specshow(mfcc_cat, x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'猫 - MFCC {i+1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MFCC系数', fontsize=10)
        
        ax = axes[i, 1]
        librosa.display.specshow(mfcc_dog, x_axis='time', sr=SR, ax=ax)
        ax.set_title(f'狗 - MFCC {i+1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MFCC系数', fontsize=10)
    
    plt.suptitle('MFCC特征对比 - 猫 vs 狗', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '04_mfcc_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        
        return {
            'mfcc_mean': mfcc_mean,
            'zcr': zcr_mean,
            'spectral_centroid': spectral_centroid_mean,
            'spectral_rolloff': spectral_rolloff_mean,
            'rms': rms_mean
        }
    except Exception as e:
        print(f"  警告: 无法处理文件 {file_path}: {e}")
        return None

def plot_feature_distribution(audio_files):
    print("正在生成特征分布可视化...")
    
    cat_features = []
    dog_features = []
    
    cat_files = (audio_files['train_cat'][:30] + audio_files['test_cat'][:15])
    dog_files = (audio_files['train_dog'][:30] + audio_files['test_dog'][:15])
    
    print("  提取猫音频特征...")
    for file in cat_files:
        features = extract_features(file)
        if features:
            cat_features.append(features)
    
    print("  提取狗音频特征...")
    for file in dog_files:
        features = extract_features(file)
        if features:
            dog_features.append(features)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    feature_names = ['zcr', 'spectral_centroid', 'spectral_rolloff', 'rms']
    feature_labels = ['过零率', '频谱质心', '频谱滚降点', 'RMS能量']
    
    for idx, (feat_name, feat_label) in enumerate(zip(feature_names, feature_labels)):
        ax = axes[idx // 3, idx % 3]
        
        cat_values = [f[feat_name] for f in cat_features]
        dog_values = [f[feat_name] for f in dog_features]
        
        data = pd.DataFrame({
            '值': cat_values + dog_values,
            '类别': ['猫'] * len(cat_values) + ['狗'] * len(dog_values)
        })
        
        sns.boxplot(data=data, x='类别', y='值', ax=ax, palette=['#FF6B6B', '#4ECDC4'])
        ax.set_title(f'{feat_label}分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('类别', fontsize=10)
        ax.set_ylabel(feat_label, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    mfcc_cat = np.array([f['mfcc_mean'] for f in cat_features])
    mfcc_dog = np.array([f['mfcc_mean'] for f in dog_features])
    
    mfcc_cat_mean = np.mean(mfcc_cat, axis=0)
    mfcc_dog_mean = np.mean(mfcc_dog, axis=0)
    
    x = np.arange(len(mfcc_cat_mean))
    width = 0.35
    
    ax.bar(x - width/2, mfcc_cat_mean, width, label='猫', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, mfcc_dog_mean, width, label='狗', color='#4ECDC4', alpha=0.8)
    ax.set_xlabel('MFCC系数索引', fontsize=10)
    ax.set_ylabel('平均值', fontsize=10)
    ax.set_title('MFCC系数均值对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    
    plt.suptitle('音频特征分布对比 - 猫 vs 狗', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '05_feature_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    return cat_features, dog_features

def generate_report(counts, durations, cat_features, dog_features):
    print("正在生成可视化报告...")
    
    report_path = os.path.join(OUTPUT_DIR, 'visualization_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("音频数据可视化分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("一、数据统计概览\n")
        f.write("-" * 60 + "\n")
        f.write(f"训练集-猫音频数量: {counts['训练集-猫']}\n")
        f.write(f"训练集-狗音频数量: {counts['训练集-狗']}\n")
        f.write(f"测试集-猫音频数量: {counts['测试集-猫']}\n")
        f.write(f"测试集-狗音频数量: {counts['测试集-狗']}\n")
        total = sum(counts.values())
        f.write(f"总音频文件数: {total}\n\n")
        
        f.write("二、音频时长统计\n")
        f.write("-" * 60 + "\n")
        if durations['猫']:
            f.write(f"猫音频平均时长: {np.mean(durations['猫']):.2f} 秒\n")
            f.write(f"猫音频时长标准差: {np.std(durations['猫']):.2f} 秒\n")
        if durations['狗']:
            f.write(f"狗音频平均时长: {np.mean(durations['狗']):.2f} 秒\n")
            f.write(f"狗音频时长标准差: {np.std(durations['狗']):.2f} 秒\n")
        f.write("\n")
        
        f.write("三、特征统计概览\n")
        f.write("-" * 60 + "\n")
        if cat_features:
            cat_zcr = np.mean([f['zcr'] for f in cat_features])
            cat_sc = np.mean([f['spectral_centroid'] for f in cat_features])
            cat_rms = np.mean([f['rms'] for f in cat_features])
            f.write(f"猫音频平均过零率: {cat_zcr:.4f}\n")
            f.write(f"猫音频平均频谱质心: {cat_sc:.2f} Hz\n")
            f.write(f"猫音频平均RMS能量: {cat_rms:.4f}\n\n")
        
        if dog_features:
            dog_zcr = np.mean([f['zcr'] for f in dog_features])
            dog_sc = np.mean([f['spectral_centroid'] for f in dog_features])
            dog_rms = np.mean([f['rms'] for f in dog_features])
            f.write(f"狗音频平均过零率: {dog_zcr:.4f}\n")
            f.write(f"狗音频平均频谱质心: {dog_sc:.2f} Hz\n")
            f.write(f"狗音频平均RMS能量: {dog_rms:.4f}\n\n")
        
        f.write("四、生成的可视化文件\n")
        f.write("-" * 60 + "\n")
        f.write("1. 01_data_statistics.png - 数据统计图表\n")
        f.write("2. 02_waveform_samples.png - 音频波形样本\n")
        f.write("3. 03_spectrogram_samples.png - 频谱分析样本\n")
        f.write("4. 04_mfcc_samples.png - MFCC特征样本\n")
        f.write("5. 05_feature_distribution.png - 特征分布对比\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"  已保存: {report_path}")

def main():
    print("\n" + "="*60)
    print("开始音频数据可视化分析")
    print("="*60 + "\n")
    
    print("步骤1: 获取音频文件列表...")
    audio_files = get_audio_files()
    for key, files in audio_files.items():
        print(f"  {key}: {len(files)} 个文件")
    
    print("\n步骤2: 生成数据统计可视化...")
    counts, durations = plot_data_statistics(audio_files)
    
    print("\n步骤3: 生成音频波形可视化...")
    plot_waveform_samples(audio_files)
    
    print("\n步骤4: 生成频谱分析可视化...")
    plot_spectrogram_samples(audio_files)
    
    print("\n步骤5: 生成MFCC特征可视化...")
    plot_mfcc_samples(audio_files)
    
    print("\n步骤6: 生成特征分布可视化...")
    cat_features, dog_features = plot_feature_distribution(audio_files)
    
    print("\n步骤7: 生成可视化报告...")
    generate_report(counts, durations, cat_features, dog_features)
    
    print("\n" + "="*60)
    print("可视化分析完成！")
    print("="*60)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
