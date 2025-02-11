import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def load_baseline_data(data_dir, experiment_name):
    """
    加载特定实验的基准数据
    
    Args:
        data_dir (str): 数据根目录
        experiment_name (str): 实验目录名
    
    Returns:
        dict: 包含steps和baseline值的字典
    """
    exp_path = os.path.join(data_dir, experiment_name)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"实验目录未找到: {exp_path}")
    
    # 获取事件文件
    event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"未找到事件文件: {exp_path}")
    
    # 加载事件数据
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()
    
    # 提取训练步数和基准值
    try:
        steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
        baseline_values = [s.value for s in event_acc.Scalars('Baseline_Loss')]
        
        return {
            'steps': np.array(steps),
            'baseline': np.array(baseline_values)
        }
    except Exception as e:
        raise ValueError(f"数据提取错误: {str(e)}")

def plot_baseline_curve(data, output_path):
    """
    绘制基准学习曲线
    
    Args:
        data (dict): 包含steps和baseline值的字典
        output_path (str): 图表保存路径
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    # 绘制主曲线
    plt.plot(data['steps'], 
            data['baseline'],
            color='#1f77b4',
            linewidth=2,
            marker='o',
            markersize=4,
            markevery=max(1, len(data['steps'])//20),
            label='baseline loss')
    
    # 设置图表标题和标签
    plt.title('HalfCheetah-v4: baseline loss curve', fontsize=14, pad=20)
    plt.xlabel('env steps', fontsize=12, labelpad=10)
    plt.ylabel('baseline loss', fontsize=12, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存至: {output_path}")

def analyze_baseline_metrics(data):
    """
    分析基准数据的关键指标
    
    Args:
        data (dict): 包含baseline值的字典
    """
    baseline_values = data['baseline']
    
    metrics = {
        'min loss': np.min(baseline_values),
        'max loss': np.max(baseline_values),
        'mean loss': np.mean(baseline_values),
        'std loss': np.std(baseline_values),
        'final loss': baseline_values[-1]
    }
    
    print("\n基准损失指标分析:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

def main():
    # 设置参数
    data_dir = "./data"
    experiment_name = "q2_pg_cheetah_baseline_HalfCheetah-v4_2025-02-10_04-01-57"
    output_dir = "analysis_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        print("正在加载实验数据...")
        data = load_baseline_data(data_dir, experiment_name)
        
        # 绘制学习曲线
        output_path = os.path.join(output_dir, "baseline_learning_curve.png")
        plot_baseline_curve(data, output_path)
        
        # 分析指标
        analyze_baseline_metrics(data)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()