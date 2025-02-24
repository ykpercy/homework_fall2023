import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_name(exp_name):
    """Parse experiment name to show parameter differences from final settings"""
    base_params = {
        'final': 'Final (b=500,dst=0.99,gae=0.98,lr=0.02)',
        't14': 'dst=0.98',
        't13': 'lr=0.01',
        't12': 'gae=0.95',
        't11': 'gae=0.99',
        't10': 'b=1000,lr=0.03',
        't9': 'lr=0.03',
        't7': 'b=400,lr=0.01',
        't6': 'b=300,lr=0.005',
        't5': 'b=300,lr=0.01',
        't4': 'b=200,lr=0.01',
        't3': 'b=500,lr=0.01',
        't2': 'b=1000,lr=0.01',
        't1': 'b=2000,lr=0.01'
    }
    
    for key, param in base_params.items():
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

def plot_learning_curves(data, output_path):
    """Plot learning curves for all experiments"""
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn')
    
    # Color scheme for different parameter types
    colors = {
        'Final': '#e74c3c',  # Red for final parameters
        'b=': '#3498db',     # Blue for batch size variations
        'lr=': '#2ecc71',    # Green for learning rate variations
        'gae=': '#9b59b6',   # Purple for GAE variations
        'dst=': '#f1c40f'    # Yellow for discount variations
    }
    
    # Plot each experiment's data
    for config, exp_data in sorted(data.items()):
        if 'Final' in config:
            color = colors['Final']
            alpha = 1.0
            linewidth = 3
        else:
            # Determine color based on the primary parameter being varied
            for param, c in colors.items():
                if param in config:
                    color = c
                    break
            else:
                color = '#95a5a6'  # Grey for undefined parameters
            alpha = 0.7
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
    plt.title('InvertedPendulum-v4: Parameter Comparison', 
              fontsize=14, pad=20)
    
    # Add target line
    plt.axhline(y=1000, color='#95a5a6', linestyle=':', 
                label='Target Return (1000)', alpha=0.5)
    
    # Set legend with two columns
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), 
              loc='upper left', frameon=True, ncol=1)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_performance_summary(data):
    """Print performance summary for experiments"""
    print("\nPerformance Summary:")
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
    # List of experiment directories (excluding t8 as requested)
    target_experiments = [
        "q2_pg_pendulum_t1_InvertedPendulum-v4_2025-02-16_03-36-00",
        "q2_pg_pendulum_t2_InvertedPendulum-v4_2025-02-16_03-40-36",
        "q2_pg_pendulum_t3_InvertedPendulum-v4_2025-02-16_03-46-06",
        "q2_pg_pendulum_t4_InvertedPendulum-v4_2025-02-16_03-48-24",
        "q2_pg_pendulum_t5_InvertedPendulum-v4_2025-02-16_03-50-42",
        "q2_pg_pendulum_t6_InvertedPendulum-v4_2025-02-16_03-52-25",
        "q2_pg_pendulum_t7_InvertedPendulum-v4_2025-02-16_03-54-30",
        "q2_pg_pendulum_t9_InvertedPendulum-v4_2025-02-16_03-58-25",
        "q2_pg_pendulum_t10_InvertedPendulum-v4_2025-02-16_04-00-16",
        "q2_pg_pendulum_t11_InvertedPendulum-v4_2025-02-16_04-08-46",
        "q2_pg_pendulum_t12_InvertedPendulum-v4_2025-02-16_04-11-47",
        "q2_pg_pendulum_t13_InvertedPendulum-v4_2025-02-16_04-18-26",
        "q2_pg_pendulum_t14_InvertedPendulum-v4_2025-02-16_04-21-37",
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
    
    output_path = os.path.join(output_dir, "inverted_pendulum_parameter_comparison.png")
    plot_learning_curves(data, output_path)
    
    print(f"\nPlot saved to: {output_path}")
    print_performance_summary(data)

if __name__ == "__main__":
    main()