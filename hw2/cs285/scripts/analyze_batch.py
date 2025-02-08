import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_config(exp_name):
    """解析实验配置"""
    config_parts = []
    if 'rtg' in exp_name:
        config_parts.append('RTG')
    if 'na' in exp_name:
        config_parts.append('NA')
    
    if not config_parts:
        return 'Vanilla PG'
    return 'PG with ' + ' + '.join(config_parts)

def load_specific_experiments(data_dir, target_experiments):
    """加载指定的实验数据"""
    data = {}
    
    for exp_name in target_experiments:
        exp_path = os.path.join(data_dir, exp_name)
        if not os.path.exists(exp_path):
            print(f"警告：未找到实验目录 {exp_name}")
            continue
            
        # 获取事件文件
        event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))[0]
        # if not event_files:
        #     continue
            
        # 使用最新的事件文件
        # latest_event_file = max(event_files, key=os.path.getctime)
        
        try:
            # 加载数据
            # event_acc = EventAccumulator(latest_event_file)
            event_acc = EventAccumulator(event_files)
            event_acc.Reload()
            
            # 提取数据
            steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
            returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')]
            
            if steps and returns:
                config_name = parse_experiment_config(exp_name)
                # print(f"{config_name} and {len(steps)} steps and {len(returns)} returns")
                data[config_name] = {
                    'steps': np.array(steps),
                    'returns': np.array(returns)
                }
                # print(f"成功加载实验数据: {config_name}")
                print(f"成功加载实验数据: {config_name} and {steps[-1]} steps")
                
        except Exception as e:
            print(f"处理文件 {event_files} 时出错: {str(e)}")
            continue
    
    return data

def plot_small_batch_comparison(data, output_path):
    """绘制小批次实验对比图"""
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn')
    
    # 设置颜色方案
    colors = {
        'Vanilla PG': '#3498db',
        'PG with RTG': '#2ecc71',
        'PG with NA': '#e74c3c',
        'PG with RTG + NA': '#9b59b6'
    }
    
    # 设置线型
    linestyles = {
        'Vanilla PG': '-',
        'PG with RTG': '--',
        'PG with NA': '-.',
        'PG with RTG + NA': ':'
    }
    
    for config, exp_data in data.items():
        plt.plot(exp_data['steps'], 
                exp_data['returns'],
                label=config,
                color=colors.get(config, '#000000'),
                linestyle=linestyles.get(config, '-'),
                marker='o',
                markersize=4,
                linewidth=2,
                alpha=0.8,
                markevery=5)  # 每5个点显示一个标记
    
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    # plt.title('CartPole-v0: Small Batch Policy Gradient Variants Comparison', 
    #           fontsize=14, pad=20)
    plt.title('CartPole-v0: Large Batch Policy Gradient Variants Comparison', 
              fontsize=14, pad=20)
    
    # 添加图例，并将其放在图表外部右侧
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局以确保图例可见
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 指定实验目录
    # target_experiments = [
    #     "q2_pg_cartpole_CartPole-v0_2025-02-08_03-18-26",
    #     "q2_pg_cartpole_na_CartPole-v0_06-02-2025_09-52-13",
    #     "q2_pg_cartpole_rtg_CartPole-v0_06-02-2025_09-49-57",
    #     "q2_pg_cartpole_rtg_na_CartPole-v0_06-02-2025_09-54-18"
    # ]

    lb_experiments = [
        "q2_pg_cartpole_lb_CartPole-v0_06-02-2025_09-56-51",
        "q2_pg_cartpole_lb_na_CartPole-v0_06-02-2025_10-08-22",
        "q2_pg_cartpole_lb_rtg_CartPole-v0_06-02-2025_10-02-53",
        "q2_pg_cartpole_lb_rtg_na_CartPole-v0_06-02-2025_10-15-13"
    ]
    
    data_dir = "./data"  # 根据实际路径调整
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载实验数据...")
    # data = load_specific_experiments(data_dir, target_experiments)
    data = load_specific_experiments(data_dir, lb_experiments)
    
    if not data:
        print("错误：未找到有效的实验数据")
        return
    
    # 生成对比图
    # output_path = os.path.join(output_dir, "small_batch_comparison_04.png")
    output_path = os.path.join(output_dir, "lb_comparison_01.png")
    plot_small_batch_comparison(data, output_path)
    
    print(f"\n图表已保存至: {output_path}")
    
    # 输出性能统计
    print("\n实验性能总结:")
    for config, exp_data in data.items():
        final_return = exp_data['returns'][-1]
        max_return = np.max(exp_data['returns'])
        mean_return = np.mean(exp_data['returns'])
        print(f"\n{config}:")
        print(f"  最终回报: {final_return:.2f}")
        print(f"  最高回报: {max_return:.2f}")
        print(f"  平均回报: {mean_return:.2f}")

if __name__ == "__main__":
    main()