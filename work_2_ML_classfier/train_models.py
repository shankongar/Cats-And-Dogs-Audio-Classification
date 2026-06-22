import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from config import *
from feature_extraction import prepare_dataset
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: LightGBM未安装，将跳过LightGBM模型")

def train_model_with_grid_search(model, param_grid, X_train, y_train, model_name):
    print(f"\n训练 {model_name}...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"  最佳参数: {best_params}")
    print(f"  交叉验证准确率: {best_score:.4f}")
    print(f"  训练时间: {train_time:.2f} 秒")
    
    model_path = os.path.join(MODEL_DIR, f'{model_name}_best.pkl')
    joblib.dump(best_model, model_path)
    print(f"  模型已保存: {model_path}")
    
    return best_model, best_params, best_score, train_time

def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'predict_time': predict_time
    }
    
    print(f"  测试集准确率: {metrics['accuracy']:.4f}")
    print(f"  测试集F1分数: {metrics['f1']:.4f}")
    if metrics['auc']:
        print(f"  测试集AUC: {metrics['auc']:.4f}")
    
    return metrics, y_pred, y_pred_proba

def train_all_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("开始模型训练")
    print("="*60)
    
    all_metrics = []
    all_predictions = {}
    all_models = {}
    
    print("\n" + "-"*60)
    print("基础模型")
    print("-"*60)
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'),
        MODEL_PARAMS['logistic_regression'],
        X_train, y_train, 'logistic_regression'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'logistic_regression')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['logistic_regression'] = (y_pred, y_pred_proba)
    all_models['logistic_regression'] = model
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=True),
        MODEL_PARAMS['svm'],
        X_train, y_train, 'svm'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'svm')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['svm'] = (y_pred, y_pred_proba)
    all_models['svm'] = model
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        KNeighborsClassifier(),
        MODEL_PARAMS['knn'],
        X_train, y_train, 'knn'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'knn')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['knn'] = (y_pred, y_pred_proba)
    all_models['knn'] = model
    
    print(f"\n训练 naive_bayes...")
    start_time = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  训练时间: {train_time:.2f} 秒")
    model_path = os.path.join(MODEL_DIR, 'naive_bayes_best.pkl')
    joblib.dump(model, model_path)
    print(f"  模型已保存: {model_path}")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'naive_bayes')
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
    metrics['cv_score'] = cv_scores.mean()
    metrics['train_time'] = train_time
    metrics['best_params'] = 'default'
    all_metrics.append(metrics)
    all_predictions['naive_bayes'] = (y_pred, y_pred_proba)
    all_models['naive_bayes'] = model
    
    print("\n" + "-"*60)
    print("集成模型")
    print("-"*60)
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        MODEL_PARAMS['random_forest'],
        X_train, y_train, 'random_forest'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'random_forest')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['random_forest'] = (y_pred, y_pred_proba)
    all_models['random_forest'] = model
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        MODEL_PARAMS['gradient_boosting'],
        X_train, y_train, 'gradient_boosting'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'gradient_boosting')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['gradient_boosting'] = (y_pred, y_pred_proba)
    all_models['gradient_boosting'] = model
    
    model, params, cv_score, train_time = train_model_with_grid_search(
        AdaBoostClassifier(random_state=RANDOM_STATE),
        MODEL_PARAMS['adaboost'],
        X_train, y_train, 'adaboost'
    )
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'adaboost')
    metrics['cv_score'] = cv_score
    metrics['train_time'] = train_time
    metrics['best_params'] = str(params)
    all_metrics.append(metrics)
    all_predictions['adaboost'] = (y_pred, y_pred_proba)
    all_models['adaboost'] = model
    
    if HAS_XGBOOST:
        print(f"\n训练 xgboost...")
        start_time = time.time()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(random_state=RANDOM_STATE, 
                            eval_metric='logloss', n_jobs=-1),
            param_grid, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        model = grid_search.best_estimator_
        params = grid_search.best_params_
        cv_score = grid_search.best_score_
        print(f"  最佳参数: {params}")
        print(f"  交叉验证准确率: {cv_score:.4f}")
        print(f"  训练时间: {train_time:.2f} 秒")
        model_path = os.path.join(MODEL_DIR, 'xgboost_best.pkl')
        joblib.dump(model, model_path)
        print(f"  模型已保存: {model_path}")
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'xgboost')
        metrics['cv_score'] = cv_score
        metrics['train_time'] = train_time
        metrics['best_params'] = str(params)
        all_metrics.append(metrics)
        all_predictions['xgboost'] = (y_pred, y_pred_proba)
        all_models['xgboost'] = model
    
    if HAS_LIGHTGBM:
        print(f"\n训练 lightgbm...")
        start_time = time.time()
        param_grid = {
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(
            lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            param_grid, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        model = grid_search.best_estimator_
        params = grid_search.best_params_
        cv_score = grid_search.best_score_
        print(f"  最佳参数: {params}")
        print(f"  交叉验证准确率: {cv_score:.4f}")
        print(f"  训练时间: {train_time:.2f} 秒")
        model_path = os.path.join(MODEL_DIR, 'lightgbm_best.pkl')
        joblib.dump(model, model_path)
        print(f"  模型已保存: {model_path}")
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, 'lightgbm')
        metrics['cv_score'] = cv_score
        metrics['train_time'] = train_time
        metrics['best_params'] = str(params)
        all_metrics.append(metrics)
        all_predictions['lightgbm'] = (y_pred, y_pred_proba)
        all_models['lightgbm'] = model
    
    return all_metrics, all_predictions, all_models

def save_metrics_to_csv(all_metrics):
    df = pd.DataFrame(all_metrics)
    columns_order = ['model_name', 'cv_score', 'accuracy', 'precision', 'recall', 
                    'f1', 'auc', 'train_time', 'predict_time', 'best_params']
    df = df[columns_order]
    df = df.sort_values('accuracy', ascending=False)
    
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n性能对比表格已保存: {csv_path}")
    
    return df

def main():
    print("\n" + "="*60)
    print("机器学习模型训练流程")
    print("="*60)
    
    X_train_path = os.path.join(RESULTS_DIR, 'X_train.npy')
    X_test_path = os.path.join(RESULTS_DIR, 'X_test.npy')
    y_train_path = os.path.join(RESULTS_DIR, 'y_train.npy')
    y_test_path = os.path.join(RESULTS_DIR, 'y_test.npy')
    
    if (os.path.exists(X_train_path) and os.path.exists(X_test_path) and
        os.path.exists(y_train_path) and os.path.exists(y_test_path)):
        print("\n从本地加载已保存的特征矩阵...")
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"  测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
    else:
        print("\n本地特征矩阵不存在，开始提取特征...")
        X_train, X_test, y_train, y_test = prepare_dataset()
    
    all_metrics, all_predictions, all_models = train_all_models(
        X_train, X_test, y_train, y_test)
    
    df_comparison = save_metrics_to_csv(all_metrics)
    
    print("\n" + "="*60)
    print("模型训练完成！")
    print("="*60)
    print("\n模型性能排名（按测试集准确率）:")
    print(df_comparison[['model_name', 'accuracy', 'f1', 'auc']].to_string(index=False))
    
    best_model_name = df_comparison.iloc[0]['model_name']
    best_accuracy = df_comparison.iloc[0]['accuracy']
    print(f"\n最佳模型: {best_model_name} (准确率: {best_accuracy:.4f})")
    
    return all_metrics, all_predictions, all_models, X_test, y_test

if __name__ == '__main__':
    main()
