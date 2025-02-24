import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_name(exp_name):
    """Parse experiment name to show learning rate variations"""
    params = {
        't3': 'lr=0.01 (b=500)',
        't9': 'lr=0.03 (b=500)',
        'tfinal': 'lr=0.02 (b=500, Final)'
    }
    
    for key, param in params.items():
        if key in exp_name:
            return param
    return 'Unknown'

def load_experiment_data(data_dir, experiment_names):
    """Load data from multiple experiments"""
    data = {}
    
    for exp_name in experiment_names:
        exp_path = os.path.join(data_dir, exp_name)
        if not os.path.exists(exp_path):
            print(f"Warning: Experiment directory not found - {exp_name}")
            continue
            
        event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))
        if not event_files:
            continue
            
        try:
            event_acc = EventAccumulator(event_files[0])
            event_acc.Reload()
            
            steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
            returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')]
            
            if steps and returns:
                config_name = parse_experiment_name(exp_name)
                data[config_name] = {
                    'steps': np.array(steps),
                    'returns': np.array(returns)
                }
                print(f"Successfully loaded: {config_name}")
                
        except Exception as e:
            print(f"Error processing {exp_name}: {str(e)}")
            continue
    
    return data

def plot_lr_comparison(data, output_path):
    """Plot learning curves focusing on learning rate variations"""
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    # Define colors for different learning rates
    colors = {
        'lr=0.01': '#3498db',  # Blue
        'lr=0.02': '#e74c3c',  # Red
        'lr=0.03': '#2ecc71'   # Green
    }
    
    # Plot each experiment's data
    for config, exp_data in sorted(data.items()):
        if 'Final' in config:
            color = colors['lr=0.02']
            alpha = 1.0
            linewidth = 3
        else:
            color = colors[f"lr={config.split('=')[1][:4]}"]
            alpha = 0.8
            linewidth = 2
            
        plt.plot(exp_data['steps'], 
                exp_data['returns'],
                label=config,
                color=color,
                linewidth=linewidth,
                alpha=alpha)
    
    # Set plot properties
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    plt.title('InvertedPendulum-v4: Learning Rate Comparison (Batch Size = 500)', 
              fontsize=14, pad=20)
    
    # Add target line
    plt.axhline(y=1000, color='#95a5a6', linestyle=':', 
                label='Target Return (1000)', alpha=0.5)
    
    # Set legend
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), 
              loc='upper left', frameon=True)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_lr_summary(data):
    """Print performance summary focusing on learning rate impact"""
    print("\nLearning Rate Comparison Summary:")
    # Sort experiments by final return
    sorted_experiments = sorted(data.items(), 
                              key=lambda x: x[1]['returns'][-1], 
                              reverse=True)
    
    for config, exp_data in sorted_experiments:
        returns = exp_data['returns']
        print(f"\n{config}:")
        print(f"  Final Return: {returns[-1]:.2f}")
        print(f"  Maximum Return: {np.max(returns):.2f}")
        print(f"  Average Return: {np.mean(returns):.2f}")
        print(f"  Standard Deviation: {np.std(returns):.2f}")

def main():
    # List of experiments focusing on learning rate
    target_experiments = [
        "q2_pg_pendulum_t3_InvertedPendulum-v4_2025-02-16_03-46-06",
        "q2_pg_pendulum_t9_InvertedPendulum-v4_2025-02-16_03-58-25",
        "q2_pg_pendulum_tfinal_InvertedPendulum-v4_2025-02-16_04-26-44"
    ]
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    data = load_experiment_data(data_dir, target_experiments)
    
    if not data:
        print("Error: No valid experiment data found")
        return
    
    output_path = os.path.join(output_dir, "inverted_pendulum_lr_comparison.png")
    plot_lr_comparison(data, output_path)
    
    print(f"\nPlot saved to: {output_path}")
    print_lr_summary(data)

if __name__ == "__main__":
    main()