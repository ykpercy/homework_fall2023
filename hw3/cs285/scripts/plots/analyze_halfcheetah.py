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

def plot_eval_returns(data, output_dir):
    """Plot evaluation return (eval_return) with emphasis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify evaluation return metrics
    eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
    
    if not eval_tags:
        print("Warning: No data found for 'eval_return' metrics")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot evaluation return metrics with enhanced visibility
    for tag in eval_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=3, color='#1f77b4', label=f"Eval: {tag}")
            
            # Add a smoothed trend line for better visualization
            if len(steps) > 5:  # Only add trend if we have enough data points
                window_size = min(10, len(steps) // 5)  # Adaptive window size
                values_smooth = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                steps_smooth = steps[window_size-1:]
                plt.plot(steps_smooth, values_smooth, linewidth=2, color='#ff7f0e', 
                         label=f"Eval (Smoothed): {tag}")
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Evaluation Returns over Training Steps", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Evaluation Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save figure
    filename = os.path.join(output_dir, "HalfCheetah_eval_returns.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation return plot to {filename}")

def plot_returns_comparison(data, output_dir):
    """Plot training and evaluation returns together but with emphasis on evaluation returns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify matching metrics
    train_tags = [tag for tag in data.keys() if 'train_return' in tag.lower()]
    eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
    
    if not train_tags and not eval_tags:
        print("Warning: No data found for metrics 'train_return' or 'eval_return'")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot training return metrics with lower emphasis
    for tag in train_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=1.5, alpha=0.6, linestyle='--', 
                    label=f"Train: {tag}")
    
    # Plot evaluation return metrics with higher emphasis
    for tag in eval_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2.5, label=f"Eval: {tag}")
    
    plt.title("Training and Evaluation Returns (Eval Highlighted)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save figure
    filename = os.path.join(output_dir, "HalfCheetah_returns_comparison_eval.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {filename}")

def main():
    # Set directories with the provided path
    logdir = './data/hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_13-04-2025_23-41-25'
    logdir = './data/hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_14-04-2025_10-55-51'
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
    
    # Plot evaluation returns with focus
    plot_eval_returns(data, output_dir)
    
    # Plot both returns with emphasis on evaluation
    plot_returns_comparison(data, output_dir)

if __name__ == "__main__":
    main()