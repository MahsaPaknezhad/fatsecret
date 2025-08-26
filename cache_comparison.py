import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import stats

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

folder = "outputs2"
experiment = "2_match_foods" #"1_extract_foods"

# Load data from all runs
def load_all_runs():
    data = {}
    
    # Find all files for each model/cache combination
    patterns = {
        'nova-micro-v1_cached': f'{folder}/{experiment}_nova-micro-v1_cached_*.json',
        'nova-micro-v1_no_cache': f'{folder}/{experiment}_nova-micro-v1_no_cache_*.json',
        'nova-lite-v1_cached': f'{folder}/{experiment}_nova-lite-v1_cached_*.json',
        'nova-lite-v1_no_cache': f'{folder}/{experiment}_nova-lite-v1_no_cache_*.json'
    }
    
    for key, pattern in patterns.items():
        files = glob.glob(pattern)
        runs = []
        for file_path in files:
            with open(file_path, 'r') as f:
                runs.append(json.load(f))
        data[key] = runs
    
    return data

data = load_all_runs()

# Extract metrics with statistical measures
metrics = {}
raw_data = {}
for key, runs in data.items():
    if not runs:
        continue
        
    model = key.split('_')[0]
    cache_status = 'cached' if 'cached' in key else 'no_cache'
    
    if model not in metrics:
        metrics[model] = {}
        raw_data[model] = {}
    
    # Extract raw values for statistical analysis
    costs = []
    latencies = []
    inputs = []
    outputs = []
    
    for run in runs:
        # Calculate totals from the list of results
        total_cost = sum(item.get('cost', 0) for item in run)/len(run)
        total_latency = sum(item.get('invocation_time', 0) for item in run)/len(run)
        total_input = sum(item.get('input_tokens', 0) for item in run)/len(run)
        total_output = sum(item.get('output_tokens', 0) for item in run)/len(run)
        
        costs.append(total_cost)
        latencies.append(total_latency)
        inputs.append(total_input)
        outputs.append(total_output)
    
    raw_data[model][cache_status] = {
        'cost': costs,
        'latency': latencies,
        'input_tokens': inputs,
        'output_tokens': outputs
    }
    
    # Calculate statistics
    metrics[model][cache_status] = {
        'cost': np.mean(costs),
        'cost_std': np.std(costs, ddof=1) if len(costs) > 1 else 0,
        'latency': np.mean(latencies),
        'latency_std': np.std(latencies, ddof=1) if len(latencies) > 1 else 0,
        'input_tokens': np.mean(inputs),
        'input_std': np.std(inputs, ddof=1) if len(inputs) > 1 else 0,
        'output_tokens': np.mean(outputs),
        'output_std': np.std(outputs, ddof=1) if len(outputs) > 1 else 0,
        'runs': len(runs)
    }

# Create scientific comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cache vs No-Cache Performance Analysis', fontsize=16, fontweight='bold')

models = list(metrics.keys())
x = np.arange(len(models))
width = 0.35

# Cost comparison with error bars
cached_costs = [metrics[model]['cached']['cost'] for model in models]
no_cache_costs = [metrics[model]['no_cache']['cost'] for model in models]
cached_cost_errs = [metrics[model]['cached']['cost_std'] for model in models]
no_cache_cost_errs = [metrics[model]['no_cache']['cost_std'] for model in models]

bars1 = ax1.bar(x - width/2, cached_costs, width, label='Cost\n(Cached)', alpha=0.8, 
                yerr=cached_cost_errs, capsize=5, color='#2E86AB')
bars2 = ax1.bar(x + width/2, no_cache_costs, width, label='Cost\n(No Cache)', alpha=0.8,
                yerr=no_cache_cost_errs, capsize=5, color='#A23B72')
ax1.set_ylabel('Cost (USD)', fontweight='bold')
ax1.set_title('Cost Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace('-', '-\n') for m in models])
ax1.legend()
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Latency comparison with error bars
cached_latency = [metrics[model]['cached']['latency'] for model in models]
no_cache_latency = [metrics[model]['no_cache']['latency'] for model in models]
cached_lat_errs = [metrics[model]['cached']['latency_std'] for model in models]
no_cache_lat_errs = [metrics[model]['no_cache']['latency_std'] for model in models]

ax2.bar(x - width/2, cached_latency, width, label='Latency\n(Cached)', alpha=0.8,
        yerr=cached_lat_errs, capsize=5, color='#2E86AB')
ax2.bar(x + width/2, no_cache_latency, width, label='Latency\n(No Cache)', alpha=0.8,
        yerr=no_cache_lat_errs, capsize=5, color='#A23B72')
ax2.set_ylabel('Latency (seconds)', fontweight='bold')
ax2.set_title('Latency Comparison', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace('-', '-\n') for m in models])
ax2.legend()

# Input tokens comparison
cached_input = [metrics[model]['cached']['input_tokens'] for model in models]
no_cache_input = [metrics[model]['no_cache']['input_tokens'] for model in models]
cached_input_errs = [metrics[model]['cached']['input_std'] for model in models]
no_cache_input_errs = [metrics[model]['no_cache']['input_std'] for model in models]

ax3.bar(x - width/2, cached_input, width, label='Input Tokens\n(Cached)', alpha=0.8,
        yerr=cached_input_errs, capsize=5, color='#2E86AB')
ax3.bar(x + width/2, no_cache_input, width, label='Input Tokens\n(No Cache)', alpha=0.8,
        yerr=no_cache_input_errs, capsize=5, color='#A23B72')
ax3.set_ylabel('Input Tokens', fontweight='bold')
ax3.set_title('Input Token Usage', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([m.replace('-', '-\n') for m in models])
ax3.legend()
ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Output tokens comparison
cached_output = [metrics[model]['cached']['output_tokens'] for model in models]
no_cache_output = [metrics[model]['no_cache']['output_tokens'] for model in models]
cached_output_errs = [metrics[model]['cached']['output_std'] for model in models]
no_cache_output_errs = [metrics[model]['no_cache']['output_std'] for model in models]

ax4.bar(x - width/2, cached_output, width, label='Output Tokens\n(Cached)', alpha=0.8,
        yerr=cached_output_errs, capsize=5, color='#2E86AB')
ax4.bar(x + width/2, no_cache_output, width, label='Output Tokens\n(No Cache)', alpha=0.8,
        yerr=no_cache_output_errs, capsize=5, color='#A23B72')
ax4.set_ylabel('Output Tokens', fontweight='bold')
ax4.set_title('Output Token Generation', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([m.replace('-', '-\n') for m in models])
ax4.legend()

plt.tight_layout()
plt.savefig('step2_cache_vs_no_cache_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Print scientific summary statistics
print("\n" + "="*80)
print("CACHE VS NO-CACHE PERFORMANCE ANALYSIS")
print("Statistical Summary (Mean ± Standard Deviation)")
print("="*80)

for model in models:
    print(f"\n{model.upper().replace('-', ' ')}:")
    print("-" * 50)
    
    cached = metrics[model]['cached']
    no_cache = metrics[model]['no_cache']
    
    print(f"Sample size: n={cached['runs']} (cached), n={no_cache['runs']} (no-cache)")
    
    # Cost analysis
    cost_savings = ((no_cache['cost'] - cached['cost']) / no_cache['cost']) * 100
    cost_effect_size = (no_cache['cost'] - cached['cost']) / np.sqrt((cached['cost_std']**2 + no_cache['cost_std']**2) / 2) if (cached['cost_std'] + no_cache['cost_std']) > 0 else 0
    
    print(f"\nCOST ANALYSIS:")
    print(f"  Cached:    ${cached['cost']:.4f} ± ${cached['cost_std']:.4f}")
    print(f"  No Cache:  ${no_cache['cost']:.4f} ± ${no_cache['cost_std']:.4f}")
    print(f"  Savings:   {cost_savings:.1f}% (Effect size: {cost_effect_size:.2f})")
    
    # Latency analysis
    latency_improvement = ((no_cache['latency'] - cached['latency']) / no_cache['latency']) * 100
    latency_effect_size = (no_cache['latency'] - cached['latency']) / np.sqrt((cached['latency_std']**2 + no_cache['latency_std']**2) / 2) if (cached['latency_std'] + no_cache['latency_std']) > 0 else 0
    
    print(f"\nLATENCY ANALYSIS:")
    print(f"  Cached:     {cached['latency']:.2f} ± {cached['latency_std']:.2f} seconds")
    print(f"  No Cache:   {no_cache['latency']:.2f} ± {no_cache['latency_std']:.2f} seconds")
    print(f"  Improvement: {latency_improvement:.1f}% (Effect size: {latency_effect_size:.2f})")
    
    # Token analysis
    input_reduction = ((no_cache['input_tokens'] - cached['input_tokens']) / no_cache['input_tokens']) * 100
    
    print(f"\nTOKEN USAGE:")
    print(f"  Input tokens (cached):    {cached['input_tokens']:,.0f} ± {cached['input_std']:,.0f}")
    print(f"  Input tokens (no cache):  {no_cache['input_tokens']:,.0f} ± {no_cache['input_std']:,.0f}")
    print(f"  Input reduction:          {input_reduction:.1f}%")
    print(f"  Output tokens (cached):   {cached['output_tokens']:,.0f} ± {cached['output_std']:,.0f}")
    print(f"  Output tokens (no cache): {no_cache['output_tokens']:,.0f} ± {no_cache['output_std']:,.0f}")
    
    # Statistical significance
    if cached['runs'] > 1 and no_cache['runs'] > 1:
        cost_t, cost_p = stats.ttest_ind(raw_data[model]['cached']['cost'], raw_data[model]['no_cache']['cost'])
        latency_t, latency_p = stats.ttest_ind(raw_data[model]['cached']['latency'], raw_data[model]['no_cache']['latency'])
        
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"  Cost difference:    t={cost_t:.3f}, p={cost_p:.4f}")
        print(f"  Latency difference: t={latency_t:.3f}, p={latency_p:.4f}")

print("\n" + "="*80)
print("LEGEND:")
print("Effect size interpretation: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)")
print("Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
print("="*80)

# Generate summary table
import pandas as pd

table_data = []
for model in models:
    cached = metrics[model]['cached']
    no_cache = metrics[model]['no_cache']
    
    cost_savings = ((no_cache['cost'] - cached['cost']) / no_cache['cost']) * 100
    latency_improvement = ((no_cache['latency'] - cached['latency']) / no_cache['latency']) * 100
    input_reduction = ((no_cache['input_tokens'] - cached['input_tokens']) / no_cache['input_tokens']) * 100
    
    table_data.append({
        'Model': model.replace('-', ' ').title(),
        'Cost\n(Cached)': f"${cached['cost']:.5f}",
        'Cost\n(No Cache)': f"${no_cache['cost']:.5f}",
        'Cost\nSavings': f"{cost_savings:.1f}%",
        'Latency\n(Cached)': f"{cached['latency']:.2f}s",
        'Latency\n(No Cache)': f"{no_cache['latency']:.2f}s",
        'Latency\nImprovement': f"{latency_improvement:.1f}%",
        'Input\nToken Reduction': f"{input_reduction:.1f}%",
        'Runs': f"{cached['runs']}"
    })

df = pd.DataFrame(table_data)
print("\n" + "="*120)
print("PERFORMANCE COMPARISON TABLE")
print("="*120)
print(df.to_string(index=False))
print("="*120)

# Plot the table
fig_table, ax_table = plt.subplots(figsize=(16, 6))
ax_table.axis('tight')
ax_table.axis('off')

table = ax_table.table(cellText=df.values, colLabels=df.columns, 
                      cellLoc='center', loc='center',
                      colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F2F2F2')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Cache vs No-Cache Performance Comparison', fontsize=16, fontweight='bold', pad=20)
plt.savefig('step2_cache_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
