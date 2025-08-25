import json
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

def analyze_experiments():
    experiments = {}
    
    for filename in os.listdir('outputs/round4'):
        if filename.endswith('.json'):
            parts = filename.replace('.json', '').split('_')
            exp_name = '_'.join(parts[:-1]).replace('3_match_sizes_batch_', '')
            run_num = int(parts[-1])
            
            with open(f'outputs/round4/{filename}', 'r') as f:
                data = json.load(f)
            
            # Calculate metrics
            total_cost = sum(item.get('cost', 0) or 0 for item in data)
            total_latency = sum(item.get('invocation_time', 0) or 0 for item in data)
            total_input = sum(item.get('input_tokens', 0) or 0 for item in data)
            total_output = sum(item.get('output_tokens', 0) or 0 for item in data)
            total_tokens = total_input + total_output
            
            if exp_name not in experiments:
                experiments[exp_name] = []
            
            experiments[exp_name].append({
                'cost': total_cost,
                'latency': total_latency,
                'input_tokens': total_input,
                'output_tokens': total_output,
                'total_tokens': total_tokens,
                'cost_per_token': total_cost / total_tokens if total_tokens > 0 else 0
            })
    
    # Generate scientific format tables as figures
    generate_summary_table(experiments)
    generate_ranking_tables(experiments)

def generate_summary_table(experiments):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    headers = ['Experiment', 'Avg Cost ($)', 'Avg Latency (s)', 'Avg Input Tokens', 'Avg Output Tokens', 'Cost/Token']
    data = []
    
    for exp_name, runs in experiments.items():
        avg_cost = mean([r['cost'] for r in runs])
        avg_latency = mean([r['latency'] for r in runs])
        avg_input = mean([r['input_tokens'] for r in runs])
        avg_output = mean([r['output_tokens'] for r in runs])
        avg_cost_per_token = mean([r['cost_per_token'] for r in runs])
        
        data.append([
            exp_name,
            f"{avg_cost:.6f}",
            f"{avg_latency:.2f}",
            f"{avg_input:.0f}",
            f"{avg_output:.0f}",
            f"{avg_cost_per_token:.2e}"
        ])
    
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Round 4 Experiment Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('experiment_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_ranking_tables(experiments):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Experiment Rankings', fontsize=16, fontweight='bold', y=0.95)
    
    rankings = [
        ('Cost (Lowest to Highest)', sorted(experiments.items(), key=lambda x: mean([r['cost'] for r in x[1]]))),
        ('Latency (Fastest to Slowest)', sorted(experiments.items(), key=lambda x: mean([r['latency'] for r in x[1]]))),
        ('Cost Efficiency (Best to Worst)', sorted(experiments.items(), key=lambda x: mean([r['cost_per_token'] for r in x[1]]))),
        ('Token Usage (Lowest to Highest)', sorted(experiments.items(), key=lambda x: mean([r['total_tokens'] for r in x[1]])))
    ]
    
    for idx, (title, ranking) in enumerate(rankings):
        ax = axes[idx//2, idx%2]
        ax.axis('off')
        
        data = []
        for i, (exp_name, runs) in enumerate(ranking, 1):
            if 'Cost' in title and 'Efficiency' not in title:
                value = f"${mean([r['cost'] for r in runs]):.6f}"
            elif 'Latency' in title:
                value = f"{mean([r['latency'] for r in runs]):.2f}s"
            elif 'Efficiency' in title:
                value = f"{mean([r['cost_per_token'] for r in runs]):.2e}"
            else:
                value = f"{mean([r['total_tokens'] for r in runs]):.0f}"
            
            data.append([str(i), exp_name, value])
        
        table = ax.table(cellText=data, colLabels=['Rank', 'Experiment', 'Value'], 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1, 2)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('experiment_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_experiments()
