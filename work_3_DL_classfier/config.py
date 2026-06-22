import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data', 'cats_dogs')
WORK_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(WORK_DIR, 'model_dl')
RESULTS_DIR = os.path.join(WORK_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

FEATURE_DIR = os.path.join(BASE_DIR, 'work_2_ML_classfier', 'results')

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_PARAMS = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.0002,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.002
}

MODEL_PARAMS = {
    'mlp': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    },
    'cnn1d': {
        'num_filters': [32, 64],
        'kernel_size': 3,
        'dropout_rate': 0.4
    },
    'rnn': {
        'hidden_size': 64,
        'num_layers': 2,
        'bidirectional': True,
        'dropout_rate': 0.3
    },
    'cnn_rnn': {
        'cnn_filters': 32,
        'rnn_hidden': 32,
        'dropout_rate': 0.3
    }
}

CLASS_LABELS = {0: '猫', 1: '狗'}
NUM_CLASSES = 2
