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

def plot_algorithms_comparison(reinforce_data, reparametrize_data, output_dir):
    """Plot comparison of evaluation returns between reinforce and reparametrize algorithms."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify evaluation return metrics for both algorithms
    reinforce_eval_tags = [tag for tag in reinforce_data.keys() if 'eval_return' in tag.lower()]
    reparametrize_eval_tags = [tag for tag in reparametrize_data.keys() if 'eval_return' in tag.lower()]
    
    if not reinforce_eval_tags or not reparametrize_eval_tags:
        print("Warning: No data found for 'eval_return' metrics in one or both algorithms")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot reinforce evaluation return metrics
    for tag in reinforce_eval_tags:
        steps = reinforce_data[tag]['steps']
        values = reinforce_data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2.5, color='#1f77b4', label=f"Reinforce: {tag}")
    
    # Plot reparametrize evaluation return metrics
    for tag in reparametrize_eval_tags:
        steps = reparametrize_data[tag]['steps']
        values = reparametrize_data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=2.5, color='#ff7f0e', label=f"Reparametrize: {tag}")
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Comparison of Evaluation Returns: Reinforce vs. Reparametrize", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Evaluation Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save figure
    filename = os.path.join(output_dir, "invPendulum_reinforce_vs_reparametrize_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved algorithm comparison plot to {filename}")

def main():
    # Set directories for both algorithms
    reinforce_logdir = './data/hw3_sac_sanity_invpendulum_reinforce_InvertedPendulum-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_13-04-2025_14-16-39'
    reparametrize_logdir = './data/hw3_sac_sanity_invpendulum_reparametrize_InvertedPendulum-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_14-04-2025_07-57-32'
    output_dir = 'tensorboard_plots'
    
    print(f"Reading reinforce data from: {reinforce_logdir}")
    reinforce_data = extract_data(reinforce_logdir)
    
    print(f"Reading reparametrize data from: {reparametrize_logdir}")
    reparametrize_data = extract_data(reparametrize_logdir)
    
    if not reinforce_data or not reparametrize_data:
        print("No data found or error reading tfevents files for one or both algorithms.")
        return
    
    # Print all available tags for both algorithms
    print("\nAvailable tags for reinforce algorithm:")
    for i, tag in enumerate(sorted(reinforce_data.keys()), 1):
        print(f"{i}. {tag}")
    
    print("\nAvailable tags for reparametrize algorithm:")
    for i, tag in enumerate(sorted(reparametrize_data.keys()), 1):
        print(f"{i}. {tag}")
    
    # Plot comparison of both algorithms' evaluation returns
    plot_algorithms_comparison(reinforce_data, reparametrize_data, output_dir)
    
    # Also generate individual plots for each algorithm if needed
    plot_individual_algorithm(reinforce_data, output_dir, "reinforce")
    plot_individual_algorithm(reparametrize_data, output_dir, "reparametrize")

def plot_individual_algorithm(data, output_dir, algorithm_name):
    """Plot evaluation return for an individual algorithm."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify evaluation return metrics
    eval_tags = [tag for tag in data.keys() if 'eval_return' in tag.lower()]
    
    if not eval_tags:
        print(f"Warning: No data found for 'eval_return' metrics in {algorithm_name} algorithm")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot evaluation return metrics
    for tag in eval_tags:
        steps = data[tag]['steps']
        values = data[tag]['values']
        if steps and values:
            plt.plot(steps, values, linewidth=3, label=f"Eval: {tag}")
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"{algorithm_name.capitalize()} Algorithm: Evaluation Returns", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Evaluation Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save figure
    filename = os.path.join(output_dir, f"invPendulum_{algorithm_name}_eval_returns.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {algorithm_name} evaluation return plot to {filename}")

if __name__ == "__main__":
    main()