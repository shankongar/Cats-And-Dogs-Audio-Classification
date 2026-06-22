import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

try:
    print("="*70)
    print("开始机器学习分类任务")
    print("="*70)
    sys.stdout.flush()
    
    print("\n步骤1: 导入模块...")
    sys.stdout.flush()
    from train_models import main as train_main
    from evaluate_models import evaluate_all_models
    import numpy as np
    import os
    from config import RESULTS_DIR
    
    print("\n步骤2: 训练模型...")
    sys.stdout.flush()
    all_metrics, all_predictions, all_models, X_test, y_test = train_main()
    
    print("\n步骤3: 加载训练数据...")
    sys.stdout.flush()
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    
    print("\n步骤4: 评估模型...")
    sys.stdout.flush()
    evaluate_all_models(all_metrics, all_predictions, all_models, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("任务完成！")
    print("="*70)
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n发生错误: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()
    sys.stdout.flush()
