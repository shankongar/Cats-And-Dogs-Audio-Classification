import os
import glob
import numpy as np
import librosa
from config import DATA_DIR, SR, N_MFCC

def extract_mfcc_sequence(file_path, n_mfcc=N_MFCC, sr=SR):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T
        return mfcc
    except Exception as e:
        print(f"  警告: 无法处理文件 {os.path.basename(file_path)}: {e}")
        return None

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

def prepare_hmm_data():
    print("\n" + "="*60)
    print("准备HMM数据（提取MFCC序列）")
    print("="*60)
    
    audio_files = load_audio_files()
    
    print(f"\n数据集统计:")
    print(f"  训练集-猫: {len(audio_files['train_cat'])} 个文件")
    print(f"  训练集-狗: {len(audio_files['train_dog'])} 个文件")
    print(f"  测试集-猫: {len(audio_files['test_cat'])} 个文件")
    print(f"  测试集-狗: {len(audio_files['test_dog'])} 个文件")
    
    train_cat_sequences = []
    train_dog_sequences = []
    test_sequences = []
    test_labels = []
    
    print("\n提取训练集MFCC序列...")
    for file_path in audio_files['train_cat']:
        mfcc_seq = extract_mfcc_sequence(file_path)
        if mfcc_seq is not None:
            train_cat_sequences.append(mfcc_seq)
    
    for file_path in audio_files['train_dog']:
        mfcc_seq = extract_mfcc_sequence(file_path)
        if mfcc_seq is not None:
            train_dog_sequences.append(mfcc_seq)
    
    print(f"  猫音频: {len(train_cat_sequences)} 个序列")
    print(f"  狗音频: {len(train_dog_sequences)} 个序列")
    
    print("\n提取测试集MFCC序列...")
    for file_path in audio_files['test_cat']:
        mfcc_seq = extract_mfcc_sequence(file_path)
        if mfcc_seq is not None:
            test_sequences.append(mfcc_seq)
            test_labels.append(0)
    
    for file_path in audio_files['test_dog']:
        mfcc_seq = extract_mfcc_sequence(file_path)
        if mfcc_seq is not None:
            test_sequences.append(mfcc_seq)
            test_labels.append(1)
    
    print(f"  测试集: {len(test_sequences)} 个序列")
    
    return train_cat_sequences, train_dog_sequences, test_sequences, np.array(test_labels)

if __name__ == '__main__':
    train_cat, train_dog, test, test_labels = prepare_hmm_data()
    print(f"\n数据准备完成!")
    print(f"  训练集猫序列形状: {[seq.shape for seq in train_cat[:3]]}")
    print(f"  训练集狗序列形状: {[seq.shape for seq in train_dog[:3]]}")
