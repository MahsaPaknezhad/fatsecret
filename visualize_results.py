import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_model_performance():
    # Read JSON files
    output_dir = Path("outputs/round2")
    data = []
    for file_path in output_dir.glob("*.json"):
        with open(file_path, 'r') as f:
            data.extend(json.load(f))
    
    # Group data by model
    models = {}
    for entry in data:
        model = entry['model']
        if model not in models:
            models[model] = {'latency': [], 'cost': [], 'input_tokens': [], 'output_tokens': []}
        
        # Only add non-None values
        if entry.get('invocation_time') is not None:
            models[model]['latency'].append(entry['invocation_time'])
        if entry.get('cost') is not None:
            models[model]['cost'].append(entry['cost'])
        if entry.get('input_tokens') is not None:
            models[model]['input_tokens'].append(entry['input_tokens'])
        if entry.get('output_tokens') is not None:
            models[model]['output_tokens'].append(entry['output_tokens'])
    
    # Filter out models with insufficient data
    valid_models = {name: metrics for name, metrics in models.items() 
                   if metrics['latency'] and metrics['cost']}
    
    if not valid_models:
        print("No valid data found")
        return
    
    # Prepare data for combined plots
    model_names = list(valid_models.keys())
    latencies = [np.mean(valid_models[name]['latency']) for name in model_names]
    latency_stds = [np.std(valid_models[name]['latency']) for name in model_names]
    costs = [np.mean(valid_models[name]['cost']) for name in model_names]
    input_tokens = [np.mean(valid_models[name]['input_tokens']) for name in model_names]
    output_tokens = [np.mean(valid_models[name]['output_tokens']) for name in model_names]
    
    # Figure 1: Combined Latency bar plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(model_names))
    ax1.bar(x_pos, latencies, yerr=latency_stds, capsize=5)
    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Model Latency Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('all_models_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Combined Cost and Tokens plot
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Bar plot for cost
    ax2.bar(x_pos, costs, alpha=0.7, label='Cost', color='skyblue')
    ax2.set_ylabel('Cost ($)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_xlabel('Models')
    
    # Secondary y-axis for tokens
    ax3 = ax2.twinx()
    ax3.plot(x_pos, input_tokens, 'ro-', label='Input Tokens', markersize=8, linewidth=2)
    ax3.plot(x_pos, output_tokens, 'go-', label='Output Tokens', markersize=8, linewidth=2)
    ax3.set_ylabel('Tokens', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    # Set x-axis labels
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_title('Model Cost and Token Usage Comparison')
    plt.tight_layout()
    plt.savefig('all_models_cost_tokens.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated combined plots for {len(valid_models)} models")

if __name__ == "__main__":
    visualize_model_performance()
