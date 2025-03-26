import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def extract_data(logdir):
    """使用 TensorBoard 的 EventAccumulator 提取 tfevents 文件中的数据"""
    data = defaultdict(lambda: {'steps': [], 'values': []})
    
    try:
        # 加载 event 文件
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        
        # 获取所有标量标签
        tags = event_acc.Tags().get('scalars', [])
        
        for tag in tags:
            # 为每个标签提取标量事件
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                data[tag]['steps'].append(event.step)
                data[tag]['values'].append(event.value)
    except Exception as e:
        print(f"读取 tfevents 文件时出错: {e}")
        return {}
    
    return data

def plot_eval_return_comparison(data_dict, output_dir):
    """绘制多个数据集的 eval_return 指标，并保存图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for dataset_name, data in data_dict.items():
        # 查找包含 eval_return 的指标（不区分大小写）
        matching_metrics = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
        
        if not matching_metrics:
            print(f"警告: {dataset_name} 中未找到 eval_return 数据")
            continue
        
        for tag in matching_metrics:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if not steps or not values:
                continue
                
            # 绘制数据，并在标签中标明数据集名称及指标名称
            plt.plot(steps, values, linewidth=2, label=f"{dataset_name} - {tag}")
    
    plt.title("Evaluation Return over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    filename = os.path.join(output_dir, "explore_eval_return_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {filename}")

def main():
    # 数据目录列表
    logdirs = [
        './data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_lr0001_25-03-2025_03-26-52',
        './data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_lr0001_softmaxExp_25-03-2025_08-23-11',
    ]
    output_dir = 'tensorboard_plots'
    
    # 从每个目录中提取数据
    data_dict = {}
    for logdir in logdirs:
        # 使用目录路径的最后部分作为数据集名称
        dataset_name = os.path.basename(logdir)
        
        print(f"正在读取数据: {logdir}")
        data = extract_data(logdir)
        
        if not data:
            print(f"{logdir} 中未找到数据或读取 tfevents 文件时出错。")
            continue
        
        data_dict[dataset_name] = data
    
    if not data_dict:
        print("未在任何指定目录中找到有效数据。")
        return
    
    # 将所有数据集的 eval_return 指标绘制在同一张图上
    plot_eval_return_comparison(data_dict, output_dir)

if __name__ == "__main__":
    main()
