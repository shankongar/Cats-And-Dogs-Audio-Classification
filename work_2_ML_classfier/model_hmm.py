import os
import time
import numpy as np
import joblib
from hmmlearn import hmm
from config import MODEL_DIR, RESULTS_DIR, RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')

class HMMClassifier:
    def __init__(self, n_components=5, covariance_type='diag', n_iter=100, random_state=RANDOM_STATE):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.hmm_cat = None
        self.hmm_dog = None
    
    def fit(self, train_cat_sequences, train_dog_sequences):
        print(f"\n训练HMM模型...")
        print(f"  隐状态数: {self.n_components}")
        print(f"  协方差类型: {self.covariance_type}")
        print(f"  最大迭代数: {self.n_iter}")
        
        start_time = time.time()
        
        print(f"\n  训练猫音频HMM...")
        X_cat = np.vstack(train_cat_sequences)
        lengths_cat = [len(seq) for seq in train_cat_sequences]
        
        self.hmm_cat = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.hmm_cat.fit(X_cat, lengths_cat)
        print(f"    完成! 猫音频样本数: {len(train_cat_sequences)}")
        
        print(f"\n  训练狗音频HMM...")
        X_dog = np.vstack(train_dog_sequences)
        lengths_dog = [len(seq) for seq in train_dog_sequences]
        
        self.hmm_dog = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.hmm_dog.fit(X_dog, lengths_dog)
        print(f"    完成! 狗音频样本数: {len(train_dog_sequences)}")
        
        train_time = time.time() - start_time
        print(f"\n  HMM训练完成! 用时: {train_time:.2f}秒")
        
        return train_time
    
    def predict(self, test_sequences):
        predictions = []
        
        for seq in test_sequences:
            try:
                log_prob_cat = self.hmm_cat.score(seq)
                log_prob_dog = self.hmm_dog.score(seq)
                
                prediction = 1 if log_prob_dog > log_prob_cat else 0
                predictions.append(prediction)
            except Exception as e:
                print(f"  警告: 预测失败: {e}")
                predictions.append(0)
        
        return np.array(predictions)
    
    def predict_proba(self, test_sequences):
        probabilities = []
        
        for seq in test_sequences:
            try:
                log_prob_cat = self.hmm_cat.score(seq)
                log_prob_dog = self.hmm_dog.score(seq)
                
                max_log_prob = max(log_prob_cat, log_prob_dog)
                prob_cat = np.exp(log_prob_cat - max_log_prob)
                prob_dog = np.exp(log_prob_dog - max_log_prob)
                
                total = prob_cat + prob_dog
                prob_dog_normalized = prob_dog / total
                
                probabilities.append(prob_dog_normalized)
            except Exception as e:
                print(f"  警告: 概率计算失败: {e}")
                probabilities.append(0.5)
        
        return np.array(probabilities)
    
    def save(self, model_path):
        model_data = {
            'hmm_cat': self.hmm_cat,
            'hmm_dog': self.hmm_dog,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter
        }
        joblib.dump(model_data, model_path)
        print(f"  HMM模型已保存: {model_path}")
    
    def load(self, model_path):
        model_data = joblib.load(model_path)
        self.hmm_cat = model_data['hmm_cat']
        self.hmm_dog = model_data['hmm_dog']
        self.n_components = model_data['n_components']
        self.covariance_type = model_data['covariance_type']
        self.n_iter = model_data['n_iter']
        print(f"  HMM模型已加载: {model_path}")

def train_hmm_model(train_cat_sequences, train_dog_sequences, 
                    test_sequences, test_labels, n_components=5):
    print("\n" + "="*60)
    print("训练HMM模型")
    print("="*60)
    
    hmm_classifier = HMMClassifier(n_components=n_components)
    train_time = hmm_classifier.fit(train_cat_sequences, train_dog_sequences)
    
    start_time = time.time()
    y_pred = hmm_classifier.predict(test_sequences)
    predict_time = time.time() - start_time
    
    y_pred_proba = hmm_classifier.predict_proba(test_sequences)
    
    model_path = os.path.join(MODEL_DIR, 'hmm_best.pkl')
    hmm_classifier.save(model_path)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'model_name': 'hmm',
        'accuracy': accuracy_score(test_labels, y_pred),
        'precision': precision_score(test_labels, y_pred),
        'recall': recall_score(test_labels, y_pred),
        'f1': f1_score(test_labels, y_pred),
        'auc': roc_auc_score(test_labels, y_pred_proba),
        'train_time': train_time,
        'predict_time': predict_time,
        'cv_score': None,
        'best_params': f'n_components={n_components}'
    }
    
    print(f"\n  测试集准确率: {metrics['accuracy']:.4f}")
    print(f"  测试集F1分数: {metrics['f1']:.4f}")
    print(f"  测试集AUC: {metrics['auc']:.4f}")
    
    return hmm_classifier, metrics, y_pred, y_pred_proba

if __name__ == '__main__':
    from feature_extraction_hmm import prepare_hmm_data
    
    train_cat, train_dog, test, test_labels = prepare_hmm_data()
    hmm_model, metrics, y_pred, y_pred_proba = train_hmm_model(
        train_cat, train_dog, test, test_labels, n_components=5)
    
    print(f"\nHMM训练完成!")
