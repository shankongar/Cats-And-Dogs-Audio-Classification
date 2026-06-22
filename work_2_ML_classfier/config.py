import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data', 'cats_dogs')
WORK_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(WORK_DIR, 'model_ml')
RESULTS_DIR = os.path.join(WORK_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

SR = 16000
N_MFCC = 13
RANDOM_STATE = 42
CV_FOLDS = 5

FEATURE_PARAMS = {
    'n_mfcc': N_MFCC,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'fmin': 0.0,
    'fmax': None
}

MODEL_PARAMS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
}

PLOT_PARAMS = {
    'figsize': (10, 8),
    'dpi': 300,
    'fontsize': 12,
    'title_fontsize': 14
}

CLASS_LABELS = {0: '猫', 1: '狗'}
