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

def plot_eval_return_comparison(data_dict, output_dir):
    """Plot eval_return metric from multiple datasets and save figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Define different shades of blue and red for different seeds
    blue_colors = ['#0000FF', '#000099', '#4169E1']  # Different blues
    red_colors = ['#FF0000', '#8B0000', '#DC143C']   # Different reds
    
    # Group by algorithm type (vanilla vs double Q)
    vanilla_data = {}
    doubleq_data = {}
    
    for dataset_name, data in data_dict.items():
        # Determine if this is vanilla DQN or Double Q-learning
        is_doubleq = 'doubleq' in dataset_name
        
        # Look for metric containing eval_return
        matching_metrics = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
        
        if not matching_metrics:
            print(f"Warning: No eval_return data found for {dataset_name}")
            continue
        
        for tag in matching_metrics:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if not steps or not values:
                continue
            
            # Store in appropriate dictionary based on algorithm type
            if is_doubleq:
                doubleq_data[dataset_name] = {'steps': steps, 'values': values}
            else:
                vanilla_data[dataset_name] = {'steps': steps, 'values': values}
    
    # Plot vanilla DQN data in different shades of blue
    blue_idx = 0
    for dataset_name, data in vanilla_data.items():
        # Extract seed number for label
        seed_info = "Seed " + dataset_name.split("seed_")[-1][0] if "seed_" in dataset_name else dataset_name
        
        plt.plot(data['steps'], data['values'], 
                 color=blue_colors[blue_idx % len(blue_colors)], 
                 alpha=0.9, 
                 linewidth=1.5, 
                 label=f"Vanilla DQN - {seed_info}")
        blue_idx += 1
    
    # Plot Double Q-learning data in different shades of red
    red_idx = 0
    for dataset_name, data in doubleq_data.items():
        # Extract seed number for label
        seed_info = "Seed " + dataset_name.split("seed")[-1][0] if "seed" in dataset_name else dataset_name
        
        plt.plot(data['steps'], data['values'], 
                 color=red_colors[red_idx % len(red_colors)], 
                 alpha=0.9, 
                 linewidth=1.5,
                 label=f"Double Q - {seed_info}")
        red_idx += 1
    
    plt.title("Evaluation Return: Vanilla DQN vs Double Q-learning (LunarLander-v2)")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "dqn_vs_doubleq_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {filename}")

def main():
    # Set directories for LunarLander-v2 experiments
    logdirs = [
        './data/hw3_dqn_dqn_LunarLander-v2_seed_1',
        './data/hw3_dqn_dqn_LunarLander-v2_seed_2',
        './data/hw3_dqn_dqn_LunarLander-v2_seed_3',
        './data/hw3_dqn_dqn_LunarLander-v2_seed1_doubleq_24-03-2025_07-50-12',
        './data/hw3_dqn_dqn_LunarLander-v2_seed2_doubleq_24-03-2025_08-18-04',
        './data/hw3_dqn_dqn_LunarLander-v2_seed3_doubleq_24-03-2025_08-41-03'
    ]
    output_dir = 'tensorboard_plots'
    
    # Extract data from all directories
    data_dict = {}
    for logdir in logdirs:
        # Use the last part of the directory path as the dataset name
        dataset_name = os.path.basename(logdir)
        
        print(f"Reading data from: {logdir}")
        data = extract_data(logdir)
        
        if not data:
            print(f"No data found or error reading tfevents files in {logdir}.")
            continue
        
        print(f"Found {len(data)} metrics in {dataset_name}")
        data_dict[dataset_name] = data
    
    if not data_dict:
        print("No valid data found in any of the specified directories.")
        return
    
    # Plot eval_return comparison
    plot_eval_return_comparison(data_dict, output_dir)

if __name__ == "__main__":
    main()