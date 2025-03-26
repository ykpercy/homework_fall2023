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
    
    for dataset_name, data in data_dict.items():
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
                
            # Plot the data with dataset name in the label
            plt.plot(steps, values, linewidth=2, label=f"{dataset_name}")
    
    plt.title("LunarLander-v2 Evaluation Return over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "lunar_lander_eval_return_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {filename}")

def main():
    # Set directories for LunarLander-v2 with different seeds
    logdirs = [
        './data/hw3_dqn_dqn_LunarLander-v2_seed_1',
        './data/hw3_dqn_dqn_LunarLander-v2_seed_2',
        './data/hw3_dqn_dqn_LunarLander-v2_seed_3'
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
        
        data_dict[dataset_name] = data
    
    if not data_dict:
        print("No valid data found in any of the specified directories.")
        return
    
    # Plot eval_return from all datasets
    plot_eval_return_comparison(data_dict, output_dir)

if __name__ == "__main__":
    main()