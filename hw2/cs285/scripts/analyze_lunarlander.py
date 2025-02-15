import os
import glob
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_name(exp_name):
    """解析实验名称以获取更简洁的显示标签，提取lambda值"""
    """解析实验配置，提取lambda值"""
    match = re.search(r'lambda_([\d\.]+)', exp_name)
    if match:
        lambda_val = match.group(1)
        return f"λ = {lambda_val}"
    else:
        return exp_name

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
        'λ = 0': '#3498db',
        'λ = 1': '#e74c3c',
        'λ = 095': '#2ecc71',
        'λ = 098': '#9b59b6',
        'λ = 099': '#f1c40f'
    }
    
    markers = {
        'λ = 0': 'o',
        'λ = 1': 's',
        'λ = 095': '^',
        'λ = 098': 'D',
        'λ = 099': 'v'
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
    plt.title('LunarLander-v2: Lambda Comparison', 
              fontsize=14, pad=20)
    
    # 添加目标线 (LunarLander-v2 达到 200 分以上算解决)
    plt.axhline(y=180, color='#95a5a6', linestyle=':', 
                label='Target Return (180+)', alpha=0.5)
    
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
        "q2_pg_lunar_lander_lambda_0_LunarLander-v2_2025-02-14_13-57-28",
        "q2_pg_lunar_lander_lambda_1_LunarLander-v2_2025-02-15_09-48-32",
        "q2_pg_lunar_lander_lambda_095_LunarLander-v2_2025-02-14_14-14-07",
        "q2_pg_lunar_lander_lambda_098_LunarLander-v2_2025-02-15_09-00-13",
        "q2_pg_lunar_lander_lambda_099_LunarLander-v2_2025-02-15_09-24-01"
    ]
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载实验数据...")
    data = load_experiment_data(data_dir, target_experiments)
    
    if not data:
        print("错误：未找到有效的实验数据")
        return
    
    output_path = os.path.join(output_dir, "lunarlander_lambda_comparison.png")
    plot_experiments_comparison(data, output_path)
    
    print(f"\n图表已保存至: {output_path}")
    print_performance_summary(data)

if __name__ == "__main__":
    main()
