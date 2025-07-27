import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def plot_csv_data(csv_file, x_column=None, x_label="X轴", y_label="Y轴", 
                  title="数据曲线图", figure_size=(12, 8), save_path=None):
    """
    绘制CSV文件的曲线图
    
    参数:
    csv_file: str - CSV文件路径
    x_column: str - 指定作为横坐标的列名，如果为None则使用行数作为横坐标
    x_label: str - X轴标签
    y_label: str - Y轴标签
    title: str - 图像标题
    figure_size: tuple - 图像大小 (宽, 高)
    save_path: str - 保存图像的路径，如果为None则只显示不保存
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(csv_file)
        print(f"成功读取CSV文件: {csv_file}")
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
        
        # 创建图像
        plt.figure(figsize=figure_size)
        
        # 确定x轴数据
        if x_column is not None:
            if x_column not in data.columns:
                raise ValueError(f"指定的列名 '{x_column}' 不存在于CSV文件中")
            x_data = data[x_column]
            # 获取除了指定x轴列之外的所有数值列
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if x_column in numeric_columns:
                numeric_columns.remove(x_column)
            plot_columns = numeric_columns
        else:
            # 使用行数作为横坐标
            x_data = range(len(data))
            # 获取所有数值列
            plot_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not plot_columns:
            raise ValueError("没有找到可以绘制的数值列")
        
        # 定义颜色列表
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_columns)))
        
        # 绘制每一列的曲线
        for i, column in enumerate(plot_columns[3:5]):
            y_data = data[column]
            # 处理缺失值
            valid_indices = ~(pd.isna(x_data) | pd.isna(y_data))
            x_valid = np.array(x_data)[valid_indices]
            y_valid = np.array(y_data)[valid_indices]
            
            plt.plot(x_valid, y_valid, color=colors[i], 
                    label=column, linewidth=2, marker='o', markersize=4)
        
        # 设置图表属性
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # 添加图例到右上角
        plt.legend(loc='upper right', frameon=True, shadow=True, 
                  fancybox=True, framealpha=0.9)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 自动调整布局
        plt.tight_layout()
        
        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_file}'")
        return False
    except pd.errors.EmptyDataError:
        print("错误: CSV文件为空")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    """
    主函数 - 示例使用方法
    """
    print(os.getcwd())
    # 文件参数配置区域 - 在使用前请修改这些参数
    csv_file_path = "../output/episode1/eval_log.csv"  # CSV文件路径
    x_column_name = 'epoch'        # 指定X轴列名，如果为None则使用行数
    x_axis_label = "epoch"       # X轴标签
    y_axis_label = "loss"       # Y轴标签
    chart_title = "损失图"   # 图表标题
    output_path = None          # 输出图像路径，如果为None则只显示不保存
    
    # 调用绘图函数
    success = plot_csv_data(
        csv_file=csv_file_path,
        x_column=x_column_name,
        x_label=x_axis_label,
        y_label=y_axis_label,
        title=chart_title,
        save_path=output_path
    )
    
    if success:
        print("绘图完成！")
    else:
        print("绘图失败，请检查参数设置和文件路径。")

if __name__ == "__main__":
    main()
    
    # 取消下面的注释来查看使用示例
    # example_usage()