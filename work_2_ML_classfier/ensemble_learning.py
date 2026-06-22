import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import joblib
from config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR, RANDOM_STATE, CV_FOLDS
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_voting_ensemble(X_train, y_train, X_test, y_test):
    print("\n训练Voting集成...")
    
    estimators = [
        ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ('svm', SVC(random_state=RANDOM_STATE, probability=True)),
        ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100))
    ]
    
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    
    voting_hard.fit(X_train, y_train)
    voting_soft.fit(X_train, y_train)
    
    y_pred_hard = voting_hard.predict(X_test)
    y_pred_soft = voting_soft.predict(X_test)
    
    acc_hard = accuracy_score(y_test, y_pred_hard)
    acc_soft = accuracy_score(y_test, y_pred_soft)
    
    print(f"  硬投票准确率: {acc_hard:.4f}")
    print(f"  软投票准确率: {acc_soft:.4f}")
    
    joblib.dump(voting_hard, os.path.join(MODEL_DIR, 'voting_hard.pkl'))
    joblib.dump(voting_soft, os.path.join(MODEL_DIR, 'voting_soft.pkl'))
    
    return voting_hard, voting_soft, acc_hard, acc_soft

def create_stacking_ensemble(X_train, y_train, X_test, y_test):
    print("\n训练Stacking集成...")
    
    estimators = [
        ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ('svm', SVC(random_state=RANDOM_STATE, probability=True)),
        ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)),
        ('gb', GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=CV_FOLDS
    )
    
    stacking.fit(X_train, y_train)
    
    y_pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  Stacking准确率: {acc:.4f}")
    
    joblib.dump(stacking, os.path.join(MODEL_DIR, 'stacking.pkl'))
    
    return stacking, acc

def analyze_ensemble_diversity(X_train, y_train, models):
    print("\n分析基模型多样性...")
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_train)
    
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))
    model_names = list(models.keys())
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            agreement = np.mean(predictions[name1] == predictions[name2])
            agreement_matrix[i, j] = agreement
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=model_names, yticklabels=model_names, ax=ax)
    ax.set_title('基模型预测一致性矩阵', fontsize=14, fontweight='bold')
    
    output_path = os.path.join(FIGURES_DIR, 'ensemble_diversity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")
    
    return agreement_matrix

def plot_ensemble_improvement(base_scores, ensemble_scores):
    print("\n绘制集成性能提升图...")
    
    all_scores = {**base_scores, **ensemble_scores}
    df = pd.DataFrame(list(all_scores.items()), columns=['model', 'accuracy'])
    df = df.sort_values('accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = []
    for model in df['model']:
        if 'voting' in model.lower() or 'stacking' in model.lower():
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')
    
    bars = ax.barh(range(len(df)), df['accuracy'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['model'])
    ax.set_xlabel('准确率', fontsize=12)
    ax.set_title('集成学习性能提升对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, df['accuracy']):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.4f}', va='center', fontsize=9)
    
    output_path = os.path.join(FIGURES_DIR, 'ensemble_improvement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

def main():
    print("\n" + "="*70)
    print("集成学习深度研究")
    print("="*70)
    
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    
    voting_hard, voting_soft, acc_hard, acc_soft = create_voting_ensemble(
        X_train, y_train, X_test, y_test)
    
    stacking, acc_stack = create_stacking_ensemble(
        X_train, y_train, X_test, y_test)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X_train, y_train),
        'SVM': SVC(random_state=RANDOM_STATE, probability=True).fit(X_train, y_train),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100).fit(X_train, y_train)
    }
    
    analyze_ensemble_diversity(X_train, y_train, models)
    
    base_scores = {
        'LogisticRegression': accuracy_score(y_test, models['LogisticRegression'].predict(X_test)),
        'SVM': accuracy_score(y_test, models['SVM'].predict(X_test)),
        'RandomForest': accuracy_score(y_test, models['RandomForest'].predict(X_test))
    }
    
    ensemble_scores = {
        'Voting (Hard)': acc_hard,
        'Voting (Soft)': acc_soft,
        'Stacking': acc_stack
    }
    
    plot_ensemble_improvement(base_scores, ensemble_scores)
    
    print("\n" + "="*70)
    print("集成学习研究完成！")
    print("="*70)

if __name__ == '__main__':
    main()
