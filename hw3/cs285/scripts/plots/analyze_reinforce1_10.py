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

def plot_combined_eval_returns(data_dict, output_dir):
    """Plot evaluation returns from multiple algorithms together for comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--']
    
    color_idx = 0
    for algorithm_name, data in data_dict.items():
        # Identify evaluation return metrics
        eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
        
        if not eval_tags:
            print(f"Warning: No data found for 'eval_return' metrics in {algorithm_name}")
            continue
        
        for tag in eval_tags:
            steps = data[tag]['steps']
            values = data[tag]['values']
            if steps and values:
                plt.plot(steps, values, linewidth=3, color=colors[color_idx % len(colors)], 
                         label=f"{algorithm_name}: {tag}",
                         linestyle=linestyles[color_idx // len(colors) % len(linestyles)])
                
                # Add a smoothed trend line for better visualization
                if len(steps) > 5:  # Only add trend if we have enough data points
                    window_size = min(10, len(steps) // 5)  # Adaptive window size
                    values_smooth = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    steps_smooth = steps[window_size-1:]
                    plt.plot(steps_smooth, values_smooth, linewidth=2, 
                             color=colors[(color_idx + 1) % len(colors)], 
                             label=f"{algorithm_name}: {tag} (Smoothed)",
                             linestyle=linestyles[color_idx // len(colors) % len(linestyles)])
                
                color_idx += 1
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Comparison of Evaluation Returns: Reinforce1 vs Reinforce10", fontsize=16, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Return", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save figure
    filename = os.path.join(output_dir, "HalfCheetah_eval_returns_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined evaluation return plot to {filename}")

def main():
    # Set directories for both algorithms
    logdir_reinforce1 = './data/hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_13-04-2025_23-41-25'
    logdir_reinforce10 = './data/hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_14-04-2025_10-55-51'
    output_dir = 'tensorboard_plots'
    
    # Dictionary to store data from both algorithms
    all_data = {}
    
    # Extract data for Reinforce1
    print(f"Reading data from Reinforce1: {logdir_reinforce1}")
    data_reinforce1 = extract_data(logdir_reinforce1)
    if data_reinforce1:
        all_data['Reinforce1'] = data_reinforce1
    else:
        print("No data found or error reading tfevents files for Reinforce1.")
    
    # Extract data for Reinforce10
    print(f"Reading data from Reinforce10: {logdir_reinforce10}")
    data_reinforce10 = extract_data(logdir_reinforce10)
    if data_reinforce10:
        all_data['Reinforce10'] = data_reinforce10
    else:
        print("No data found or error reading tfevents files for Reinforce10.")
    
    if not all_data:
        print("No data found for either algorithm.")
        return
    
    # Print available tags for each algorithm
    for algo_name, data in all_data.items():
        print(f"\nAvailable tags for {algo_name}:")
        for i, tag in enumerate(sorted(data.keys()), 1):
            print(f"{i}. {tag}")
    
    # Plot evaluation returns from both algorithms together
    plot_combined_eval_returns(all_data, output_dir)
    
    # Optional: You can still generate individual plots as in the original code
    # for algo_name, data in all_data.items():
    #     plot_eval_returns(data, output_dir, suffix=f"_{algo_name}")

if __name__ == "__main__":
    main()