import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_name(exp_name):
    """解析实验名称以获取更简洁的显示标签"""
    if 'bgs1' in exp_name:
        return 'Baseline (Batch Size 1)'
    elif 'blr0001' in exp_name:
        return 'Baseline (LR 0.001)'
    elif 'baseline_na' in exp_name:
        return 'Baseline (No Advantage)'
    elif 'baseline' in exp_name and not any(x in exp_name for x in ['bgs1', 'blr0001', 'na']):
        return 'Baseline (Default)'
    else:
        return 'No Baseline'

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
    """绘制多个实验的对比图"""
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    # 定义不同实验的颜色和线型
    colors = {
        'No Baseline': '#3498db',
        'Baseline (Default)': '#e74c3c',
        'Baseline (Batch Size 1)': '#2ecc71',
        'Baseline (LR 0.001)': '#9b59b6',
        'Baseline (No Advantage)': '#f1c40f'
    }
    
    markers = {
        'No Baseline': 'o',
        'Baseline (Default)': 's',
        'Baseline (Batch Size 1)': '^',
        'Baseline (LR 0.001)': 'D',
        'Baseline (No Advantage)': 'v'
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
                markevery=len(exp_data['steps'])//20)
    
    # 设置图表属性
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    plt.title('HalfCheetah-v4: Policy Gradient Variants Comparison', 
              fontsize=14, pad=20)
    
    # 添加目标线
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
    # 实验目录列表
    target_experiments = [
        "q2_pg_cheetah_baseline_bgs1_HalfCheetah-v4_2025-02-10_04-26-54",
        "q2_pg_cheetah_baseline_blr0001_HalfCheetah-v4_2025-02-10_07-59-19",
        "q2_pg_cheetah_baseline_HalfCheetah-v4_2025-02-10_04-01-57",
        "q2_pg_cheetah_baseline_na_HalfCheetah-v4_2025-02-10_08-36-40",
        "q2_pg_cheetah_HalfCheetah-v4_2025-02-10_03-54-52"
    ]
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载实验数据...")
    data = load_experiment_data(data_dir, target_experiments)
    
    if not data:
        print("错误：未找到有效的实验数据")
        return
    
    output_path = os.path.join(output_dir, "halfcheetah_variants_comparison.png")
    plot_experiments_comparison(data, output_path)
    
    print(f"\n图表已保存至: {output_path}")
    print_performance_summary(data)

if __name__ == "__main__":
    main()