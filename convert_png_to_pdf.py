# -*- coding: utf-8 -*-
"""
统一将三个work文件夹中的PNG图片转换为PDF文件
用于LaTeX实验报告的图片引用
"""

import os
from PIL import Image
import sys


def convert_single_png_to_pdf(png_path, pdf_path, dpi=300):
    """
    将单个PNG文件转换为PDF
    
    Args:
        png_path: PNG文件路径
        pdf_path: 输出PDF路径
        dpi: 分辨率（默认300）
    
    Returns:
        bool: 转换是否成功
    """
    try:
        img = Image.open(png_path)
        
        if img.mode == 'RGBA':
            # RGBA模式需要转换为RGB，使用白色背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(pdf_path, 'PDF', resolution=dpi)
        
        return True
    except Exception as e:
        print(f"  [错误] {png_path}: {e}")
        return False


def convert_folder_png_to_pdf(folder_path):
    """
    递归遍历文件夹，将所有PNG转换为PDF
    
    Args:
        folder_path: 要扫描的文件夹路径
    
    Returns:
        tuple: (成功数量, 失败数量, 总数)
    """
    success_count = 0
    fail_count = 0
    total_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_path = os.path.join(root, file)
                pdf_path = os.path.splitext(png_path)[0] + '.pdf'
                
                total_count += 1
                
                # 如果PDF已存在且比PNG新，则跳过
                if os.path.exists(pdf_path) and \
                   os.path.getmtime(pdf_path) >= os.path.getmtime(png_path):
                    success_count += 1
                    continue
                
                if convert_single_png_to_pdf(png_path, pdf_path):
                    success_count += 1
                else:
                    fail_count += 1
    
    return success_count, fail_count, total_count


def main():
    """
    主函数：处理三个work文件夹的PNG转PDF
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 定义三个需要处理的文件夹
    folders = [
        ('任务一：数据可视化', os.path.join(project_root, 'work_1_visualize')),
        ('任务二：传统机器学习', os.path.join(project_root, 'work_2_ML_classfier')),
        ('任务三：深度学习', os.path.join(project_root, 'work_3_dl_classfier')),
    ]
    
    print("=" * 70)
    print("PNG → PDF 批量转换工具")
    print("=" * 70)
    print(f"\n项目根目录: {project_root}\n")
    
    total_success = 0
    total_fail = 0
    total_files = 0
    
    for folder_name, folder_path in folders:
        if not os.path.exists(folder_path):
            print(f"[跳过] {folder_name} - 文件夹不存在: {folder_path}")
            continue
        
        print(f"📁 处理 {folder_name}")
        print(f"   路径: {folder_path}")
        
        success, fail, total = convert_folder_png_to_pdf(folder_path)
        
        total_success += success
        total_fail += fail
        total_files += total
        
        print(f"   结果: 成功 {success}/{total}, 失败 {fail}/{total}\n")
    
    print("=" * 70)
    print("📊 总体统计")
    print("=" * 70)
    print(f"   处理文件总数: {total_files}")
    print(f"   转换成功:     {total_success} ✅")
    print(f"   转换失败:     {total_fail} ❌")
    print(f"   成功率:       {(total_success/total_files*100):.1f}%" if total_files > 0 else "   无文件处理")
    print("=" * 70)
    
    if total_fail > 0:
        print("\n⚠️  部分文件转换失败，请检查上方错误信息")
        return 1
    else:
        print("\n✅ 所有PNG文件已成功转换为PDF！")
        return 0


if __name__ == '__main__':
    sys.exit(main())
