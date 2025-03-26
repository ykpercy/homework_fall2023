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

def plot_returns(data, output_dir):
    """Plot average training return (train_return) and evaluation return (eval_return) on the same axes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify matching metrics
    train_tags = [tag for tag in data.keys() if 'train_return' in tag.lower()]
    eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
    
    if not train_tags and not eval_tags:
        print("Warning: No data found for metrics 'train_return' or 'eval_return'")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot training return metrics
    for tag in train_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2, label=f"Train: {tag}")
    
    # Plot evaluation return metrics
    for tag in eval_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2, label=f"Eval: {tag}")
    
    plt.title("Training and Evaluation Returns over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "pacman_returns.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    # Set directories in main function
    logdir = './data/hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_24-03-2025_09-20-52'
    output_dir = 'tensorboard_plots'
    
    print(f"Reading data from: {logdir}")
    data = extract_data(logdir)
    
    if not data:
        print("No data found or error reading tfevents files.")
        return
    
    print(f"Found {len(data)} metrics: {', '.join(data.keys())}")
    
    # Plot both training and evaluation returns
    plot_returns(data, output_dir)

if __name__ == "__main__":
    main()
