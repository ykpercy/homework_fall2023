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

def plot_q_value_analysis(data, output_dir):
    """Plot Q-values and target values for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Specifically analyze q_values and target_values
    q_metrics = ['q_values', 'target_values']
    
    plt.figure(figsize=(12, 6))
    for tag in q_metrics:
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if steps and values:
                plt.plot(steps, values, linewidth=2, label=tag)
    
    plt.title("Q-Values and Target Values over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "q_value_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Q-value analysis plot to {filename}")
    
    # Calculate and display the difference between q_values and target_values
    if 'q_values' in data and 'target_values' in data:
        q_steps = data['q_values']['steps']
        q_vals = data['q_values']['values']
        t_steps = data['target_values']['steps']
        t_vals = data['target_values']['values']
        
        # Ensure we're comparing at the same steps
        common_indices = set(q_steps).intersection(t_steps)
        if common_indices:
            plt.figure(figsize=(12, 6))
            
            # Get values at common indices
            differences = []
            common_steps = []
            
            for i, step in enumerate(q_steps):
                if step in t_steps:
                    t_idx = t_steps.index(step)
                    diff = abs(q_vals[i] - t_vals[t_idx])
                    differences.append(diff)
                    common_steps.append(step)
            
            plt.plot(common_steps, differences, linewidth=2, color='purple')
            plt.title("Difference between Q-Values and Target Values over Training")
            plt.xlabel("Training Steps")
            plt.ylabel("Absolute Difference")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            filename = os.path.join(output_dir, "q_target_difference.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Q-value/target difference plot to {filename}")

def plot_critic_error(data, output_dir):
    """Plot critic error and related metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot critic loss
    if 'critic_loss' in data:
        plt.figure(figsize=(12, 6))
        steps = data['critic_loss']['steps']
        values = data['critic_loss']['values']
        
        if steps and values:
            plt.plot(steps, values, linewidth=2, color='red')
            
            # Add exponential moving average for clearer trend visualization
            window_size = min(50, len(values) // 10) if len(values) > 50 else 5
            if window_size > 0:
                smoothed_values = []
                alpha = 2 / (window_size + 1)
                smoothed = values[0]
                smoothed_values.append(smoothed)
                
                for i in range(1, len(values)):
                    smoothed = alpha * values[i] + (1 - alpha) * smoothed
                    smoothed_values.append(smoothed)
                
                plt.plot(steps, smoothed_values, linewidth=2.5, color='darkred', 
                         linestyle='--', label='Smoothed Trend')
        
        plt.title("Critic Loss over Training Steps")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(['Critic Loss', 'Smoothed Trend'])
        
        # Save figure
        filename = os.path.join(output_dir, "critic_loss.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved critic loss plot to {filename}")

def plot_performance_metrics(data, output_dir):
    """Plot performance metrics (returns) to relate to Q-values and critic error."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and evaluation returns together
    plt.figure(figsize=(12, 6))
    
    metrics_to_plot = ['train_return', 'eval_return']
    for tag in metrics_to_plot:
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            
            if steps and values:
                plt.plot(steps, values, linewidth=2, label=tag)
    
    plt.title("Performance (Returns) over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    filename = os.path.join(output_dir, "performance_metrics.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance metrics plot to {filename}")

def plot_training_parameters(data, output_dir):
    """Plot training parameters that affect learning."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot epsilon (exploration rate)
    if 'epsilon' in data:
        plt.figure(figsize=(12, 6))
        steps = data['epsilon']['steps']
        values = data['epsilon']['values']
        
        if steps and values:
            plt.plot(steps, values, linewidth=2, color='green')
            
        plt.title("Exploration Rate (Epsilon) over Training Steps")
        plt.xlabel("Training Steps")
        plt.ylabel("Epsilon Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        filename = os.path.join(output_dir, "epsilon.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved epsilon plot to {filename}")

def calculate_correlation(data):
    """Calculate correlation between critic loss, q-values and performance."""
    correlations = []
    
    # Check if we have both critic_loss and eval_return
    if 'critic_loss' in data and 'eval_return' in data:
        loss_steps = set(data['critic_loss']['steps'])
        eval_steps = set(data['eval_return']['steps'])
        
        # Find common steps
        common_steps = loss_steps.intersection(eval_steps)
        
        if common_steps:
            # Extract values at common steps
            loss_values = []
            eval_values = []
            
            for step in sorted(common_steps):
                loss_idx = data['critic_loss']['steps'].index(step)
                eval_idx = data['eval_return']['steps'].index(step)
                
                loss_values.append(data['critic_loss']['values'][loss_idx])
                eval_values.append(data['eval_return']['values'][eval_idx])
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(loss_values, eval_values)[0, 1]
            correlations.append(("Critic Loss vs Eval Return", correlation))
    
    # Check if we have both q_values and eval_return
    if 'q_values' in data and 'eval_return' in data:
        q_steps = set(data['q_values']['steps'])
        eval_steps = set(data['eval_return']['steps'])
        
        # Find common steps
        common_steps = q_steps.intersection(eval_steps)
        
        if common_steps:
            # Extract values at common steps
            q_values = []
            eval_values = []
            
            for step in sorted(common_steps):
                q_idx = data['q_values']['steps'].index(step)
                eval_idx = data['eval_return']['steps'].index(step)
                
                q_values.append(data['q_values']['values'][q_idx])
                eval_values.append(data['eval_return']['values'][eval_idx])
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(q_values, eval_values)[0, 1]
            correlations.append(("Q-Values vs Eval Return", correlation))
    
    return correlations

def print_metric_statistics(data):
    """Print statistics for key metrics."""
    key_metrics = ['q_values', 'target_values', 'critic_loss', 'eval_return', 'train_return']
    
    print("\n==== Metric Statistics ====")
    for tag in key_metrics:
        if tag in data and data[tag]['values']:
            values = data[tag]['values']
            print(f"Metric: {tag}")
            print(f"  Min: {min(values):.4f}")
            print(f"  Max: {max(values):.4f}")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std Dev: {np.std(values):.4f}")
            
            # Calculate trends (early vs late)
            if len(values) > 10:
                early_avg = np.mean(values[:len(values)//5])
                late_avg = np.mean(values[-len(values)//5:])
                change_pct = ((late_avg - early_avg) / early_avg * 100) if early_avg != 0 else float('inf')
                print(f"  Early avg: {early_avg:.4f}, Late avg: {late_avg:.4f}")
                print(f"  Change: {change_pct:.2f}%")
            
            print(f"  Final value: {values[-1]:.4f}")
            print()

def main():
    # Set directories in main function
    # logdir = './data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_22-03-2025_03-34-39'
    logdir = './data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_19-03-2025_04-18-18'
    output_dir = 'tensorboard_plots/cartpole_lr0001'
    
    print(f"Reading data from: {logdir}")
    data = extract_data(logdir)
    
    if not data:
        print("No data found or error reading tfevents files.")
        return
    
    print(f"Found {len(data)} metrics: {', '.join(data.keys())}")
    
    # Generate all analysis plots
    plot_q_value_analysis(data, output_dir)
    plot_critic_error(data, output_dir)
    plot_performance_metrics(data, output_dir)
    plot_training_parameters(data, output_dir)
    
    # Calculate correlations
    correlations = calculate_correlation(data)
    if correlations:
        print("\n==== Correlations ====")
        for pair, corr in correlations:
            print(f"{pair}: {corr:.4f}")
    
    # Print statistics for all metrics
    print_metric_statistics(data)
    
    print("\n==== Analysis Summary ====")
    print("The plots and statistics above show the relationship between:")
    print("1. Q-values and target values (convergence of approximation)")
    print("2. Critic loss (TD error being minimized during training)")
    print("3. Performance metrics (how the agent's policy improves)")
    print("4. Exploration parameters (epsilon decay affecting learning)")
    print("\nThese metrics relate to key DQN concepts like temporal difference learning,")
    print("function approximation, and the exploration-exploitation trade-off.")

if __name__ == "__main__":
    main()