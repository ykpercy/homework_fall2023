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
    filename = os.path.join(output_dir, "pendulum_returns_test5.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filename}")

def plot_entropy(data, output_dir):
    """Plot entropy and actor_entropy curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify entropy metrics
    entropy_tags = [tag for tag in data.keys() if 'entropy' in tag.lower()]
    # target_values_tags = [tag for tag in data.keys() if 'target_values' in tag.lower()]
    
    if not entropy_tags:
        print("Warning: No data found for entropy metrics")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot entropy metrics
    for tag in entropy_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2, label=tag)
    
    # # Plot target values metrics
    # for tag in target_values_tags:
    #     steps = data[tag]['steps']
    #     values = data[tag]['values']
    #     if steps and values:
    #         plt.plot(steps, values, linewidth=2, label=tag)
    
    plt.title("Entropy Metrics over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Entropy Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add a horizontal line for log(2) ≈ 0.693
    plt.axhline(y=np.log(2), color='r', linestyle='--', label='log(2) ≈ 0.693')
    
    # Save figure
    filename = os.path.join(output_dir, "entropy_curves_test5.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    # Set directories with the provided path
    # logdir = './data/hw3_sac_sanity_pendulum_Pendulum-v1_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_htu1000_30-03-2025_03-53-29'
    # logdir = './data/hw3_sac_sanity_pendulum_Pendulum-v1_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_02-04-2025_10-28-08'
    logdir = './data/hw3_sac_sanity_pendulum_Pendulum-v1_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_13-04-2025_13-08-27'
    output_dir = 'tensorboard_plots'
    
    print(f"Reading data from: {logdir}")
    data = extract_data(logdir)
    
    if not data:
        print("No data found or error reading tfevents files.")
        return
    
    # Print all available tags
    print("Available tags:")
    for i, tag in enumerate(sorted(data.keys()), 1):
        print(f"{i}. {tag}")
    
    # Plot returns
    plot_returns(data, output_dir)
    
    # Plot entropy metrics
    plot_entropy(data, output_dir)

if __name__ == "__main__":
    main()