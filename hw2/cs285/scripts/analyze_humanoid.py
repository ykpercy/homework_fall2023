import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def load_experiment_data(data_dir, experiment_name):
    """Load experiment data for the first 300 iterations"""
    exp_path = os.path.join(data_dir, experiment_name)
    if not os.path.exists(exp_path):
        print(f"Warning: Experiment directory not found {experiment_name}")
        return None
        
    event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))[0]
    
    try:
        event_acc = EventAccumulator(event_files)
        event_acc.Reload()
        
        # steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')][:300]
        # returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')][:300]
        steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
        returns = [s.value for s in event_acc.Scalars('Eval_AverageReturn')]
        
        if steps and returns:
            print(f"Successfully loaded data: iterations: {len(steps)}, steps: {steps[-1]}")
            return {
                'steps': np.array(steps),
                'returns': np.array(returns)
            }
            
    except Exception as e:
        print(f"Error processing {event_files}: {str(e)}")
    
    return None

def plot_learning_curve(data, output_path):
    """Plot Humanoid learning curve"""
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn')
    
    plt.plot(data['steps'], 
            data['returns'],
            color='#3498db',
            marker='o',
            markersize=4,
            linewidth=2,
            alpha=0.8,
            markevery=10,
            label='Policy Gradient')
    
    plt.xlabel('Environment Steps', fontsize=12, labelpad=10)
    plt.ylabel('Average Return', fontsize=12, labelpad=10)
    plt.title('Humanoid-v4: Policy Gradient Learning Curve', 
              fontsize=14, pad=20)
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # experiment_name = "q2_pg_humanoid_Humanoid-v4_2025-02-18_09-54-03"
    experiment_name = "q2_pg_humanoid_acc_Humanoid-v4_2025-03-03_03-47-45"
    
    data_dir = "./data"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    data = load_experiment_data(data_dir, experiment_name)
    
    if data is None:
        print("Error: No valid experiment data found")
        return
    
    output_path = os.path.join(output_dir, "humanoid_learning_curve_e1000.png")
    plot_learning_curve(data, output_path)
    
    print(f"\nPlot saved to: {output_path}")
    
    print("\nPerformance Summary:")
    final_return = data['returns'][-1]
    max_return = np.max(data['returns'])
    mean_return = np.mean(data['returns'])
    print(f"  Final Return: {final_return:.2f}")
    print(f"  Max Return: {max_return:.2f}")
    print(f"  Mean Return: {mean_return:.2f}")

if __name__ == "__main__":
    main()