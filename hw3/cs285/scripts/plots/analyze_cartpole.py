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

def plot_eval_return(data, output_dir):
    """Plot eval_return metric and save figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for metric containing eval_return
    matching_metrics = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
    
    if not matching_metrics:
        print("Warning: No data found for metric 'eval_return'")
        return
    
    plt.figure(figsize=(12, 6))
    
    for tag in matching_metrics:
        steps = data[tag]['steps']
        values = data[tag]['values']
        
        if not steps or not values:
            continue
            
        # Plot the data
        plt.plot(steps, values, linewidth=2, label=tag)
    
    plt.title("Evaluation Return over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "eval_return.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    # Set directories in main function
    logdir = 'data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_18-03-2025_09-21-02'
    output_dir = 'tensorboard_plots'
    
    print(f"Reading data from: {logdir}")
    data = extract_data(logdir)
    
    if not data:
        print("No data found or error reading tfevents files.")
        return
    
    print(f"Found {len(data)} metrics: {', '.join(data.keys())}")
    
    # Only plot eval_return
    plot_eval_return(data, output_dir)

if __name__ == "__main__":
    main()