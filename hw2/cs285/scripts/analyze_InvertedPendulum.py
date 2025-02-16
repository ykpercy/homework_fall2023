import os
import glob
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_name(exp_name):
    """解析实验名称以提取种子信息"""
    # 假设种子以 _s数字_ 的格式出现在名称中
    match = re.search(r'_s(\d+)_', exp_name)
    if match:
        seed = match.group(1)
        return f'Seed {seed}'
    else:
        return 'Unknown Seed'

def load_experiment_data(data_dir, experiment_names):
    """加载多个实验的数据"""
    data = {}
    
    for exp_name in experiment_names:
        exp_path = os.path.join(data_dir, exp_name)
        if not os.path.exists(exp_path):
            print(f"警告：未找到实验目录 {exp_name}")
            continue
            
        event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))
        if not event_files:
            print(f"警告：在 {exp_name} 中未找到事件文件")
            continue
            
        try:
            event_acc = EventAccumulator(event_files[0])
            event_acc.Reload()
            
            steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
            returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')]
            
            if steps and returns:
                config_name = parse_experiment_name(exp_name)
                data[config_name] = {
                    'steps': np.array(steps),
                    'returns': np.array(returns)
                }
                print(f"成功加载: {config_name}")
                
        except Exception as e:
            print(f"处理 {exp_name} 时出错: {str(e)}")
            continue
    
    return data

def plot_experiments_comparison(data, output_path):
    """绘制多个实验的种子对比图"""
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    # 定义不同种子的颜色和标记
    colors = {
        'Seed 1': '#3498db',
        'Seed 2': '#e74c3c',
        'Seed 3': '#2ecc71',
        'Seed 4': '#9b59b6',
        'Seed 5': '#f1c40f'
    }
    
    markers = {
        'Seed 1': 'o',
        'Seed 2': 's',
        'Seed 3': '^',
        'Seed 4': 'D',
        'Seed 5': 'v'
    }
    
    # 绘制每个实验的数据
    for config, exp_data in data.items():
        plt.plot(exp_data['steps'], 
                 exp_data['returns'],
                 label=config,
                 color=colors.get(config, '#000000'),
                 marker=markers.get(config, 'o'),
                 markersize=6,
                 linewidth=2,
                 alpha=0.8,
                 markevery=max(1, len(exp_data['steps'])//20))
    
    # 设置图表属性
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    plt.title('InvertedPendulum-v4: Policy Gradient Default Seed Comparison', 
              fontsize=14, pad=20)
    
    # 如果有目标回报线，可以选择添加（此处示例为300）
    plt.axhline(y=300, color='#95a5a6', linestyle=':', 
                label='Target Return (300)', alpha=0.5)
    
    # 设置图例
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), 
               loc='upper left', frameon=True)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_performance_summary(data):
    """打印实验性能总结"""
    print("\n实验性能总结:")
    for config, exp_data in data.items():
        returns = exp_data['returns']
        print(f"\n{config}:")
        print(f"  最终回报: {returns[-1]:.2f}")
        print(f"  最大回报: {np.max(returns):.2f}")
        print(f"  平均回报: {np.mean(returns):.2f}")
        print(f"  标准差: {np.std(returns):.2f}")

def main():
    # 实验目录列表（种子对比）
    target_experiments = [
        "q2_pg_pendulum_default_s1_InvertedPendulum-v4_2025-02-15_14-03-43",
        "q2_pg_pendulum_default_s2_InvertedPendulum-v4_2025-02-15_14-10-23",
        "q2_pg_pendulum_default_s3_InvertedPendulum-v4_2025-02-15_14-18-27",
        "q2_pg_pendulum_default_s4_InvertedPendulum-v4_2025-02-15_14-24-53",
        "q2_pg_pendulum_default_s5_InvertedPendulum-v4_2025-02-15_14-31-11"
    ]
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载实验数据...")
    data = load_experiment_data(data_dir, target_experiments)
    
    if not data:
        print("错误：未找到有效的实验数据")
        return
    
    output_path = os.path.join(output_dir, "invertedpendulum_seed_comparison.png")
    plot_experiments_comparison(data, output_path)
    
    print(f"\n图表已保存至: {output_path}")
    print_performance_summary(data)

if __name__ == "__main__":
    main()
