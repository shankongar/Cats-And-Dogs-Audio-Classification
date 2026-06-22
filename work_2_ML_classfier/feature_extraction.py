import os
import glob
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
from config import *
import warnings
warnings.filterwarnings('ignore')

def extract_features_from_audio(y, sr=SR):
    features = []
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                                n_fft=FEATURE_PARAMS['n_fft'],
                                hop_length=FEATURE_PARAMS['hop_length'])
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))
    
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.append(np.mean(spectral_flatness))
    features.append(np.std(spectral_flatness))
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spectral_contrast, axis=1))
    features.extend(np.std(spectral_contrast, axis=1))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    return np.array(features)

def load_audio_files():
    train_cat = glob.glob(os.path.join(DATA_DIR, 'train', 'cat', '*.wav'))
    train_dog = glob.glob(os.path.join(DATA_DIR, 'train', 'dog', '*.wav'))
    test_cat = glob.glob(os.path.join(DATA_DIR, 'test', 'cats', '*.wav'))
    test_dog = glob.glob(os.path.join(DATA_DIR, 'test', 'test', '*.wav'))
    
    if not train_cat:
        train_cat = glob.glob(os.path.join(DATA_DIR, 'cat_*.wav'))[:125]
    if not train_dog:
        train_dog = glob.glob(os.path.join(DATA_DIR, 'dog_barking_*.wav'))[:85]
    if not test_cat:
        test_cat = glob.glob(os.path.join(DATA_DIR, 'cat_*.wav'))[125:164]
    if not test_dog:
        test_dog = glob.glob(os.path.join(DATA_DIR, 'dog_barking_*.wav'))[85:113]
    
    return {
        'train_cat': train_cat,
        'train_dog': train_dog,
        'test_cat': test_cat,
        'test_dog': test_dog
    }

def extract_features_from_files(file_list, label):
    features_list = []
    labels_list = []
    
    for file_path in file_list:
        try:
            y, sr = librosa.load(file_path, sr=SR)
            features = extract_features_from_audio(y, sr)
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"  警告: 无法处理文件 {os.path.basename(file_path)}: {e}")
    
    return features_list, labels_list

def prepare_dataset():
    print("\n" + "="*60)
    print("开始特征提取")
    print("="*60)
    
    audio_files = load_audio_files()
    
    print(f"\n数据集统计:")
    print(f"  训练集-猫: {len(audio_files['train_cat'])} 个文件")
    print(f"  训练集-狗: {len(audio_files['train_dog'])} 个文件")
    print(f"  测试集-猫: {len(audio_files['test_cat'])} 个文件")
    print(f"  测试集-狗: {len(audio_files['test_dog'])} 个文件")
    
    print("\n提取训练集特征...")
    train_cat_features, train_cat_labels = extract_features_from_files(
        audio_files['train_cat'], 0)
    train_dog_features, train_dog_labels = extract_features_from_files(
        audio_files['train_dog'], 1)
    
    X_train = np.array(train_cat_features + train_dog_features)
    y_train = np.array(train_cat_labels + train_dog_labels)
    
    print(f"  训练集特征维度: {X_train.shape}")
    
    print("\n提取测试集特征...")
    test_cat_features, test_cat_labels = extract_features_from_files(
        audio_files['test_cat'], 0)
    test_dog_features, test_dog_labels = extract_features_from_files(
        audio_files['test_dog'], 1)
    
    X_test = np.array(test_cat_features + test_dog_features)
    y_test = np.array(test_cat_labels + test_dog_labels)
    
    print(f"  测试集特征维度: {X_test.shape}")
    
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  标准化器已保存: {scaler_path}")
    
    print("\n保存处理好的特征矩阵...")
    np.save(os.path.join(RESULTS_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(RESULTS_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(RESULTS_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'), y_test)
    print(f"  特征矩阵已保存到: {RESULTS_DIR}")
    
    print(f"\n最终数据集:")
    print(f"  训练集: {X_train_scaled.shape[0]} 样本, {X_train_scaled.shape[1]} 特征")
    print(f"  测试集: {X_test_scaled.shape[0]} 样本, {X_test_scaled.shape[1]} 特征")
    print(f"  类别分布 - 训练集: 猫={np.sum(y_train==0)}, 狗={np.sum(y_train==1)}")
    print(f"  类别分布 - 测试集: 猫={np.sum(y_test==0)}, 狗={np.sum(y_test==1)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_dataset()
    print("\n特征提取完成！")
