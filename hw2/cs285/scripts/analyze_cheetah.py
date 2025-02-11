import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def parse_experiment_config(exp_name):
    """Parse experiment configuration"""
    if 'baseline' in exp_name:
        return 'HalfCheetah with Baseline'
    return 'HalfCheetah without Baseline'

def load_specific_experiments(data_dir, target_experiments):
    """Load specified experiment data"""
    data = {}
    
    for exp_name in target_experiments:
        exp_path = os.path.join(data_dir, exp_name)
        if not os.path.exists(exp_path):
            print(f"Warning: Experiment directory not found {exp_name}")
            continue
            
        event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))[0]
        
        try:
            event_acc = EventAccumulator(event_files)
            event_acc.Reload()
            
            steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
            returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')]
            
            if steps and returns:
                config_name = parse_experiment_config(exp_name)
                data[config_name] = {
                    'steps': np.array(steps),
                    'returns': np.array(returns)
                }
                print(f"Successfully loaded: {config_name}, steps: {steps[-1]}")
                
        except Exception as e:
            print(f"Error processing {event_files}: {str(e)}")
            continue
    
    return data

def plot_cheetah_comparison(data, output_path):
    """Plot HalfCheetah experiment comparison"""
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn')
    
    colors = {
        'HalfCheetah without Baseline': '#3498db',
        'HalfCheetah with Baseline': '#e74c3c'
    }
    
    linestyles = {
        'HalfCheetah without Baseline': '-',
        'HalfCheetah with Baseline': '--'
    }
    
    for config, exp_data in data.items():
        plt.plot(exp_data['steps'], 
                exp_data['returns'],
                label=config,
                color=colors.get(config, '#000000'),
                linestyle=linestyles.get(config, '-'),
                marker='o',
                markersize=4,
                linewidth=2,
                alpha=0.8,
                markevery=5)
    
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    plt.title('HalfCheetah-v4: Policy Gradient with/without Baseline', 
              fontsize=14, pad=20)
    
    # Add target line at 300
    plt.axhline(y=300, color='g', linestyle=':', label='Target Return (300)', alpha=0.5)
    
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    target_experiments = [
        "q2_pg_cheetah_HalfCheetah-v4_2025-02-10_03-54-52",
        "q2_pg_cheetah_baseline_HalfCheetah-v4_2025-02-10_04-01-57"
    ]
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    data = load_specific_experiments(data_dir, target_experiments)
    
    if not data:
        print("Error: No valid experiment data found")
        return
    
    output_path = os.path.join(output_dir, "halfcheetah_comparison.png")
    plot_cheetah_comparison(data, output_path)
    
    print(f"\nPlot saved to: {output_path}")
    
    print("\nExperiment Performance Summary:")
    for config, exp_data in data.items():
        final_return = exp_data['returns'][-1]
        max_return = np.max(exp_data['returns'])
        mean_return = np.mean(exp_data['returns'])
        print(f"\n{config}:")
        print(f"  Final Return: {final_return:.2f}")
        print(f"  Max Return: {max_return:.2f}")
        print(f"  Mean Return: {mean_return:.2f}")

if __name__ == "__main__":
    main()