import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize TensorBoard data from tfevents files')
    parser.add_argument('--logdir', type=str, 
                        default='data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_18-03-2025_09-21-02',
                        help='Path to the directory containing tfevents files')
    parser.add_argument('--output', type=str, default='tensorboard_plots',
                        help='Directory to save the generated plots')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['train_return', 'eval_return', 'epsilon', 'loss'],
                        help='Metrics to plot')
    parser.add_argument('--smooth', type=int, default=10,
                        help='Window size for smoothing')
    return parser.parse_args()

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

def smooth_data(values, window_size):
    """Apply moving average smoothing to the data."""
    if window_size <= 1 or len(values) <= window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    
    return smoothed

def plot_metrics(data, metrics, output_dir, smooth_window):
    """Plot selected metrics and save figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        matching_metrics = [tag for tag in data.keys() if metric.lower() in tag.lower()]
        
        if not matching_metrics:
            print(f"Warning: No data found for metric '{metric}'")
            continue
        
        plt.figure(figsize=(12, 6))
        
        for tag in matching_metrics:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if not steps or not values:
                continue
                
            # Plot raw data with low alpha
            plt.plot(steps, values, alpha=0.3, label=f"{tag} (raw)")
            
            # Plot smoothed data
            smoothed_values = smooth_data(values, smooth_window)
            plt.plot(steps, smoothed_values, linewidth=2, label=f"{tag} (smoothed)")
        
        plt.title(f"{metric} over Training Steps")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save figure
        filename = os.path.join(output_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {filename}")

def create_summary_plot(data, output_dir, smooth_window):
    """Create a summary plot with key metrics for reinforcement learning."""
    plt.figure(figsize=(15, 10))
    
    # Important metrics for DQN
    key_metrics = ['eval_return', 'train_return', 'epsilon', 'loss']
    for i, metric in enumerate(key_metrics):
        plt.subplot(2, 2, i+1)
        
        matching_metrics = [tag for tag in data.keys() if metric.lower() in tag.lower()]
        for tag in matching_metrics:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if not steps or not values:
                continue
                
            smoothed_values = smooth_data(values, smooth_window)
            plt.plot(steps, smoothed_values, linewidth=2, label=tag)
            
        plt.title(f"{metric}")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "dqn_training_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {summary_path}")

def main():
    args = parse_args()
    
    print(f"Reading data from: {args.logdir}")
    data = extract_data(args.logdir)
    
    if not data:
        print("No data found or error reading tfevents files.")
        return
    
    print(f"Found {len(data)} metrics: {', '.join(data.keys())}")
    
    # Plot requested metrics
    plot_metrics(data, args.metrics, args.output, args.smooth)
    
    # Generate a summary plot
    create_summary_plot(data, args.output, args.smooth)

if __name__ == "__main__":
    main()