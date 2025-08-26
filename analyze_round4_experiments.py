import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean, stdev

def load_experiment_data(folder_path):
    experiments = {}
    
    for filename in sorted(os.listdir(folder_path)):
        print(filename)
        if filename.endswith('.json'):
            parts = filename.replace('.json', '').split('_')
            exp_name = '_'.join(parts[:-1])
            run_num = int(parts[-1])
            
            if exp_name not in experiments:
                experiments[exp_name] = {}
            
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                
            # Aggregate metrics for this run
            total_cost = sum(item.get('cost', 0) or 0 for item in data)/len(data)
            total_latency = sum(item.get('invocation_time', 0) or 0 for item in data)/len(data)
            total_input_tokens = sum(item.get('input_tokens', 0) or 0 for item in data)/len(data)
            total_output_tokens = sum(item.get('output_tokens', 0) or 0 for item in data)/len(data)
            
            experiments[exp_name][run_num] = {
                'total_cost': total_cost,
                'total_latency': total_latency,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'num_requests': len(data)
            }
    
    return experiments

def create_comparison_plots(experiments):
    # Set scientific plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Prepare data for plotting with specific order
    desired_order = ['old', 'new', 'ultra']
    exp_names = []
    costs = []
    latencies = []
    input_tokens = []
    output_tokens = []
    
    # Sort experiments by desired order
    sorted_experiments = []
    for order_name in desired_order:
        for exp_name, runs in experiments.items():
            clean_name = exp_name.replace('3_match_sizes_batch_', '')
            if clean_name == order_name:
                sorted_experiments.append((clean_name, runs))
                break
    
    for exp_name, runs in sorted_experiments:
        exp_names.append(exp_name)
        
        run_costs = [run['total_cost'] for run in runs.values()]
        run_latencies = [run['total_latency'] for run in runs.values()]
        run_input_tokens = [run['total_input_tokens'] for run in runs.values()]
        run_output_tokens = [run['total_output_tokens'] for run in runs.values()]
        
        costs.append(run_costs)
        latencies.append(run_latencies)
        input_tokens.append(run_input_tokens)
        output_tokens.append(run_output_tokens)
    
    # Create subplots with scientific styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experimental Performance Comparison Across Batch Configurations', fontsize=16, y=0.98)
    
    # Cost comparison with error bars
    means_cost = [np.mean(c) for c in costs]
    stds_cost = [np.std(c) for c in costs]
    bp1 = ax1.boxplot(costs, labels=exp_names, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax1.errorbar(range(1, len(means_cost)+1), means_cost, yerr=stds_cost, 
                fmt='ko', capsize=5, capthick=2, label='Mean ± SD')
    ax1.set_ylabel('Total Cost (USD)', fontweight='bold')
    ax1.set_title('(A) Cost Analysis', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Latency comparison with error bars
    means_lat = [np.mean(l) for l in latencies]
    stds_lat = [np.std(l) for l in latencies]
    bp2 = ax2.boxplot(latencies, labels=exp_names, patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax2.errorbar(range(1, len(means_lat)+1), means_lat, yerr=stds_lat, 
                fmt='ko', capsize=5, capthick=2, label='Mean ± SD')
    ax2.set_ylabel('Total Latency (seconds)', fontweight='bold')
    ax2.set_title('(B) Latency Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Input tokens comparison with error bars
    means_in = [np.mean(i) for i in input_tokens]
    stds_in = [np.std(i) for i in input_tokens]
    bp3 = ax3.boxplot(input_tokens, labels=exp_names, patch_artist=True,
                      boxprops=dict(facecolor='lightyellow', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax3.errorbar(range(1, len(means_in)+1), means_in, yerr=stds_in, 
                fmt='ko', capsize=5, capthick=2, label='Mean ± SD')
    ax3.set_ylabel('Total Input Tokens', fontweight='bold')
    ax3.set_xlabel('Batch Configuration', fontweight='bold')
    ax3.set_title('(C) Input Token Usage', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Output tokens comparison with error bars
    means_out = [np.mean(o) for o in output_tokens]
    stds_out = [np.std(o) for o in output_tokens]
    bp4 = ax4.boxplot(output_tokens, labels=exp_names, patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax4.errorbar(range(1, len(means_out)+1), means_out, yerr=stds_out, 
                fmt='ko', capsize=5, capthick=2, label='Mean ± SD')
    ax4.set_ylabel('Total Output Tokens', fontweight='bold')
    ax4.set_xlabel('Batch Configuration', fontweight='bold')
    ax4.set_title('(D) Output Token Usage', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('round4_experiment_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def print_summary_stats(experiments):
    results = []
    
    for exp_name, runs in experiments.items():
        costs = [run['total_cost'] for run in runs.values()]
        latencies = [run['total_latency'] for run in runs.values()]
        input_tokens = [run['total_input_tokens'] for run in runs.values()]
        output_tokens = [run['total_output_tokens'] for run in runs.values()]
        
        result = {
            'experiment': exp_name.replace('3_match_sizes_batch_', ''),
            'avg_cost': mean(costs),
            'std_cost': stdev(costs) if len(costs) > 1 else 0,
            'avg_latency': mean(latencies),
            'std_latency': stdev(latencies) if len(latencies) > 1 else 0,
            'avg_input_tokens': mean(input_tokens),
            'std_input_tokens': stdev(input_tokens) if len(input_tokens) > 1 else 0,
            'avg_output_tokens': mean(output_tokens),
            'std_output_tokens': stdev(output_tokens) if len(output_tokens) > 1 else 0,
            'num_runs': len(runs)
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    print("=== EXPERIMENT SUMMARY STATISTICS ===\n")
    print("COST:")
    cost_df = df[['experiment', 'avg_cost', 'std_cost']].sort_values('avg_cost')
    for _, row in cost_df.iterrows():
        print(f"  {row['experiment']}: ${row['avg_cost']:.6f} (±${row['std_cost']:.6f})")
    
    print("\nLATENCY:")
    latency_df = df[['experiment', 'avg_latency', 'std_latency']].sort_values('avg_latency')
    for _, row in latency_df.iterrows():
        print(f"  {row['experiment']}: {row['avg_latency']:.2f}s (±{row['std_latency']:.2f}s)")
    
    print("\nINPUT TOKENS:")
    input_df = df[['experiment', 'avg_input_tokens', 'std_input_tokens']].sort_values('avg_input_tokens')
    for _, row in input_df.iterrows():
        print(f"  {row['experiment']}: {row['avg_input_tokens']:.0f} (±{row['std_input_tokens']:.0f})")
    
    print("\nOUTPUT TOKENS:")
    output_df = df[['experiment', 'avg_output_tokens', 'std_output_tokens']].sort_values('avg_output_tokens')
    for _, row in output_df.iterrows():
        print(f"  {row['experiment']}: {row['avg_output_tokens']:.0f} (±{row['std_output_tokens']:.0f})")

# Main execution
folder_path = '/home/ubuntu/projects/fatsecret/outputs3/round4'
experiments = load_experiment_data(folder_path)

print_summary_stats(experiments)
create_comparison_plots(experiments)
