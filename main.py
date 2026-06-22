import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def run_script(script_path, cwd=None):
    if cwd is None:
        cwd = PROJECT_ROOT
    
    script_path = Path(script_path)
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    
    if not script_path.exists():
        print(f"错误：脚本不存在 - {script_path}")
        return False
    
    print(f"\n执行: {script_path.relative_to(PROJECT_ROOT)}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            check=False
        )
        
        if result.returncode != 0:
            print(f"\n警告：脚本执行返回非零状态码 {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"\n执行出错: {e}")
        return False

def print_menu():
    print("\n" + "="*70)
    print("音频分类实验 - 主控制台")
    print("="*70)
    print(f"\n项目根目录: {PROJECT_ROOT}")
    print("\n请选择要执行的任务：\n")
    
    print("【基础任务】")
    print("  1. 任务一：数据可视化")
    print("  2. 任务二：传统机器学习分类")
    print("  3. 任务三：深度神经网络分类")
    
    print("\n【优化扩展】")
    print("  4. 深度特征分析（PCA、t-SNE）")
    print("  5. HMM模型训练")
    print("  6. 模型解释性分析（SHAP）")
    print("  7. 集成学习研究")
    print("  8. 错误分析")
    print("  9. 超参数自动调优（Optuna）")
    print("  10. 模型变体实验（Attention/Residual/BiLSTM/GRU）")
    
    print("\n【综合分析】")
    print("  11. 跨任务综合对比分析")
    print("  12. 生成LaTeX实验报告")
    
    print("\n【自创新模型】")
    print("  16. MS-TFANet - 多尺度时频注意力网络")
    
    print("\n【一键运行】")
    print("  17. 运行所有基础任务")
    print("  18. 运行所有优化扩展")
    print("  19. 运行完整实验流程")
    
    print("\n  0. 退出")
    print("\n" + "="*70)

def run_task_1():
    print("\n" + "="*70)
    print("执行任务一：数据可视化")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_1_visualize' / 'visualize_audio.py'
    run_script(script, cwd=script.parent)

def run_task_2():
    print("\n" + "="*70)
    print("执行任务二：传统机器学习分类")
    print("="*70)
    
    ml_dir = PROJECT_ROOT / 'work_2_ML_classfier'
    
    train_script = ml_dir / 'train_models.py'
    eval_script = ml_dir / 'evaluate_models.py'
    
    success = True
    if not run_script(train_script, cwd=ml_dir):
        success = False
    if not run_script(eval_script, cwd=ml_dir):
        success = False
    
    if success:
        print("\n任务二完成！")
    else:
        print("\n任务二执行过程中出现错误！")

def run_task_3():
    print("\n" + "="*70)
    print("执行任务三：深度神经网络分类")
    print("="*70)
    
    dl_dir = PROJECT_ROOT / 'work_3_DL_classfier'
    
    train_script = dl_dir / 'train_all.py'
    eval_script = dl_dir / 'evaluate_models.py'
    
    success = True
    if not run_script(train_script, cwd=dl_dir):
        success = False
    if not run_script(eval_script, cwd=dl_dir):
        success = False
    
    if success:
        print("\n任务三完成！")
    else:
        print("\n任务三执行过程中出现错误！")

def run_deep_feature_analysis():
    print("\n" + "="*70)
    print("执行深度特征分析")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_1_visualize' / 'deep_feature_analysis.py'
    run_script(script, cwd=script.parent)

def run_hmm():
    print("\n" + "="*70)
    print("执行HMM模型训练")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_2_ML_classfier' / 'model_hmm.py'
    run_script(script, cwd=script.parent)

def run_interpretability():
    print("\n" + "="*70)
    print("执行模型解释性分析")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_2_ML_classfier' / 'model_interpretability.py'
    run_script(script, cwd=script.parent)

def run_ensemble():
    print("\n" + "="*70)
    print("执行集成学习研究")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_2_ML_classfier' / 'ensemble_learning.py'
    run_script(script, cwd=script.parent)

def run_error_analysis():
    print("\n" + "="*70)
    print("执行错误分析")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_2_ML_classfier' / 'error_analysis.py'
    run_script(script, cwd=script.parent)

def run_hyperparameter_tuning():
    print("\n" + "="*70)
    print("执行超参数自动调优")
    print("="*70)
    
    script = PROJECT_ROOT / 'work_3_DL_classfier' / 'hyperparameter_tuning.py'
    run_script(script, cwd=script.parent)

def run_model_variants():
    print("\n" + "="*70)
    print("执行模型变体实验")
    print("="*70)
    print("\n将训练以下模型变体：")
    print("  1. MLP with Attention - 带注意力机制的MLP")
    print("  2. CNN with Residual - 带残差连接的CNN")
    print("  3. BiLSTM - 双向LSTM")
    print("  4. GRU - 门控循环单元")
    print("="*70)
    
    dl_dir = PROJECT_ROOT / 'work_3_DL_classfier'
    
    train_script = dl_dir / 'train_all.py'
    eval_script = dl_dir / 'evaluate_models.py'
    
    success = True
    if not run_script(train_script, cwd=dl_dir):
        success = False
    if not run_script(eval_script, cwd=dl_dir):
        success = False
    
    if success:
        print("\n模型变体实验完成！")
        print("\n模型文件保存在: work_3_DL_classfier/model_dl/")
        print("  - mlp_attention_best.pth")
        print("  - cnn_residual_best.pth")
        print("  - bilstm_best.pth")
        print("  - gru_best.pth")
    else:
        print("\n模型变体实验执行过程中出现错误！")

def run_comprehensive_analysis():
    print("\n" + "="*70)
    print("执行跨任务综合对比分析")
    print("="*70)
    
    script = PROJECT_ROOT / 'comprehensive_analysis.py'
    run_script(script, cwd=PROJECT_ROOT)

def run_latex_report():
    print("\n" + "="*70)
    print("生成LaTeX实验报告")
    print("="*70)
    
    script = PROJECT_ROOT / 'generate_latex_report.py'
    run_script(script, cwd=PROJECT_ROOT)

def run_ms_tfanet():
    print("\n" + "="*70)
    print("执行 MS-TFANet - 多尺度时频注意力网络")
    print("="*70)
    print("\n模型特点：")
    print("  1. 多尺度并行处理 - 同时捕获不同时间尺度的特征")
    print("  2. 时频双域注意力 - 分别在时间域和频率域学习注意力")
    print("  3. 自适应特征融合 - 动态学习不同尺度特征的贡献度")
    print("  4. 轻量化设计 - 参数效率高，适合小数据集")
    print("="*70)
    
    self_improve_dir = PROJECT_ROOT / 'self_Improve'
    
    train_script = self_improve_dir / 'train.py'
    eval_script = self_improve_dir / 'evaluate.py'
    
    success = True
    if not run_script(train_script, cwd=self_improve_dir):
        success = False
    if not run_script(eval_script, cwd=self_improve_dir):
        success = False
    
    if success:
        print("\nMS-TFANet 训练和评估完成！")
        print("\n模型文件保存在: self_Improve/model/")
        print("结果文件保存在: self_Improve/results/")
    else:
        print("\nMS-TFANet 执行过程中出现错误！")

def run_all_basic():
    print("\n" + "="*70)
    print("运行所有基础任务")
    print("="*70)
    
    print("\n>>> 任务一：数据可视化")
    run_task_1()
    
    print("\n>>> 任务二：传统机器学习分类")
    run_task_2()
    
    print("\n>>> 任务三：深度神经网络分类")
    run_task_3()
    
    print("\n" + "="*70)
    print("所有基础任务完成！")
    print("="*70)

def run_all_optimizations():
    print("\n" + "="*70)
    print("运行所有优化扩展")
    print("="*70)
    
    print("\n>>> 深度特征分析")
    run_deep_feature_analysis()
    
    print("\n>>> HMM模型训练")
    run_hmm()
    
    print("\n>>> 模型解释性分析")
    run_interpretability()
    
    print("\n>>> 集成学习研究")
    run_ensemble()
    
    print("\n>>> 错误分析")
    run_error_analysis()
    
    print("\n>>> 超参数自动调优")
    run_hyperparameter_tuning()
    
    print("\n>>> 模型变体实验")
    run_model_variants()
    
    print("\n" + "="*70)
    print("所有优化扩展完成！")
    print("="*70)

def run_full_experiment():
    print("\n" + "="*70)
    print("运行完整实验流程")
    print("="*70)
    
    print("\n【阶段1：基础任务】")
    run_all_basic()
    
    print("\n【阶段2：优化扩展】")
    run_all_optimizations()
    
    print("\n【阶段3：综合分析】")
    run_comprehensive_analysis()
    run_latex_report()
    
    print("\n" + "="*70)
    print("完整实验流程执行完毕！")
    print("="*70)
    print("\n实验结果位置：")
    print(f"  - 数据可视化: {PROJECT_ROOT / 'work_1_visualize'}")
    print(f"  - 机器学习: {PROJECT_ROOT / 'work_2_ML_classfier' / 'results'}")
    print(f"  - 深度学习: {PROJECT_ROOT / 'work_3_DL_classfier' / 'results'}")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("欢迎使用音频分类实验系统")
    print("="*70)
    
    while True:
        print_menu()
        
        try:
            choice = input("\n请输入选项编号: ").strip()
            
            if choice == '0':
                print("\n感谢使用！再见！")
                break
            elif choice == '1':
                run_task_1()
            elif choice == '2':
                run_task_2()
            elif choice == '3':
                run_task_3()
            elif choice == '4':
                run_deep_feature_analysis()
            elif choice == '5':
                run_hmm()
            elif choice == '6':
                run_interpretability()
            elif choice == '7':
                run_ensemble()
            elif choice == '8':
                run_error_analysis()
            elif choice == '9':
                run_hyperparameter_tuning()
            elif choice == '10':
                run_model_variants()
            elif choice == '11':
                run_comprehensive_analysis()
            elif choice == '12':
                run_latex_report()
            elif choice == '16':
                run_ms_tfanet()
            elif choice == '17':
                run_all_basic()
            elif choice == '18':
                run_all_optimizations()
            elif choice == '19':
                run_full_experiment()
            else:
                print("\n无效选项，请重新选择！")
                continue
            
            input("\n按回车键继续...")
            
        except KeyboardInterrupt:
            print("\n\n程序已中断！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键继续...")

if __name__ == '__main__':
    main()
