import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def extract_data(logdir):
    """Extract data from tfevents files using TensorBoard's EventAccumulator."""
    data = defaultdict(lambda: {'steps': [], 'values': []})
    
    try:
        # Load the event file
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        
        # Get all scalar tags
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            # Extract scalar events for each tag
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                data[tag]['steps'].append(event.step)
                data[tag]['values'].append(event.value)
    except Exception as e:
        print(f"Error reading tfevents file: {e}")
        return {}
    
    return data

def plot_eval_returns_comparison(data_dict, output_dir):
    """Plot evaluation returns from multiple methods for comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '-', '-', '-', '-']
    
    for i, (method_name, data) in enumerate(data_dict.items()):
        # Find eval_return metric
        eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
        
        if not eval_tags:
            print(f"Warning: No 'eval_return' metrics found for {method_name}")
            continue
            
        # Use the first eval_return tag found
        tag = eval_tags[0]
        steps = data[tag]['steps']
        values = data[tag]['values']
        
        if steps and values:
            plt.plot(steps, values, label=f"{method_name}", 
                     color=colors[i % len(colors)], 
                     linestyle=linestyles[i % len(linestyles)],
                     linewidth=2.5)
            
            # Add smoothed trend line
            if len(steps) > 5:
                window_size = min(10, len(steps) // 5)
                values_smooth = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                steps_smooth = steps[window_size-1:]
                plt.plot(steps_smooth, values_smooth, 
                         color=colors[i % len(colors)], 
                         linestyle='--',
                         linewidth=1.5,
                         alpha=0.7,
                         label=f"{method_name} (Smoothed)")
    
    plt.title("Evaluation Returns: Comparison of Q-Learning Methods", fontsize=16, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Return", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save figure
    filename = os.path.join(output_dir, "hopper_eval_returns_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation returns comparison plot to {filename}")

def plot_q_values_comparison(data_dict, output_dir):
    """Plot Q-values from multiple methods for comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '-', '-', '-', '-']
    
    for i, (method_name, data) in enumerate(data_dict.items()):
        # Find Q-value metrics
        q_tags = [tag for tag in data.keys() if 'q_val' in tag.lower() or 'q_value' in tag.lower()]
        
        if not q_tags:
            print(f"Warning: No Q-value metrics found for {method_name}")
            continue
            
        # Use the first Q-value tag found
        tag = q_tags[0]
        steps = data[tag]['steps']
        values = data[tag]['values']
        
        if steps and values:
            plt.plot(steps, values, label=f"{method_name}", 
                     color=colors[i % len(colors)], 
                     linestyle=linestyles[i % len(linestyles)],
                     linewidth=2.5)
            
            # Add smoothed trend line
            if len(steps) > 5:
                window_size = min(10, len(steps) // 5)
                values_smooth = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                steps_smooth = steps[window_size-1:]
                plt.plot(steps_smooth, values_smooth, 
                         color=colors[i % len(colors)], 
                         linestyle='--',
                         linewidth=1.5,
                         alpha=0.7,
                         label=f"{method_name} (Smoothed)")
    
    plt.title("Q-Values: Comparison of Q-Learning Methods", fontsize=16, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Q-Value", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save figure
    filename = os.path.join(output_dir, "hopper_q_values_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Q-values comparison plot to {filename}")

def main():
    # Define data directories for each method
    logdirs = {
        "Single-Q": './data/hw3_sac_sac_hopper_singlecritic_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_21-04-2025_03-12-29',
        "Double-Q": './data/hw3_sac_sac_hopper_doubleq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_doubleq_21-04-2025_03-27-59',
        "Clipped Double-Q (mean)": './data/hw3_sac_sac_hopper_clipq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_21-04-2025_09-18-38',
        "Clipped Double-Q (min)": './data/hw3_sac_sac_hopper_clipq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_min_21-04-2025_03-47-32',
        "REDQ": './data/hw3_sac_sac_hopper_clipq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_redq_21-04-2025_04-07-56'
    }
    
    output_dir = 'tensorboard_plots'
    
    # Extract data from each log directory
    all_data = {}
    for method_name, logdir in logdirs.items():
        print(f"Reading data for {method_name} from: {logdir}")
        data = extract_data(logdir)
        
        if not data:
            print(f"No data found or error reading tfevents files for {method_name}")
            continue
            
        all_data[method_name] = data
        
        # Print available tags for each method
        print(f"Available tags for {method_name}:")
        for i, tag in enumerate(sorted(data.keys()), 1):
            print(f"{i}. {tag}")
        print("---")
    
    # Generate comparison plots
    if all_data:
        plot_eval_returns_comparison(all_data, output_dir)
        plot_q_values_comparison(all_data, output_dir)
    else:
        print("No data available for plotting.")

if __name__ == "__main__":
    main()