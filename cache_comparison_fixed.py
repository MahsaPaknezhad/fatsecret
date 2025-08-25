import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
files = {
    'nova-micro-v1_cached': 'outputs/1_extract_foods_nova-micro-v1_cached.json',
    'nova-micro-v1_no_cache': 'outputs/1_extract_foods_nova-micro-v1_no_cache.json',
    'nova-lite-v1_cached': 'outputs/1_extract_foods_nova-lite-v1_cached.json',
    'nova-lite-v1_no_cache': 'outputs/1_extract_foods_nova-lite-v1_no_cache.json'
}

data = {}
for key, file_path in files.items():
    with open(file_path, 'r') as f:
        data[key] = json.load(f)

# Calculate totals for each experiment
def calculate_totals(results_array):
    total_cost = sum(item['cost'] for item in results_array)
    total_latency = sum(item['invocation_time'] for item in results_array)
    total_input_tokens = sum(item['input_tokens'] for item in results_array)
    total_output_tokens = sum(item['output_tokens'] for item in results_array)
    
    return {
        'cost': total_cost,
        'latency': total_latency,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'num_requests': len(results_array)
    }

# Extract metrics
metrics = {}
for key, content in data.items():
    model = key.split('_')[0]
    cache_status = 'cached' if 'cached' in key else 'no_cache'
    
    if model not in metrics:
        metrics[model] = {}
    
    metrics[model][cache_status] = calculate_totals(content)

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

models = list(metrics.keys())
x = np.arange(len(models))
width = 0.35

# Cost comparison
cached_costs = [metrics[model]['cached']['cost'] for model in models]
no_cache_costs = [metrics[model]['no_cache']['cost'] for model in models]

bars1 = ax1.bar(x - width/2, cached_costs, width, label='Cached', alpha=0.8, color='green')
bars2 = ax1.bar(x + width/2, no_cache_costs, width, label='No Cache', alpha=0.8, color='red')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Total Cost Comparison: Cache vs No Cache')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_costs, no_cache_costs)):
    ax1.text(i - width/2, cached + max(cached_costs) * 0.01, f'${cached:.5f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, no_cache + max(no_cache_costs) * 0.01, f'${no_cache:.5f}', ha='center', va='bottom', fontsize=9)

# Latency comparison
cached_latency = [metrics[model]['cached']['latency'] for model in models]
no_cache_latency = [metrics[model]['no_cache']['latency'] for model in models]

ax2.bar(x - width/2, cached_latency, width, label='Cached', alpha=0.8, color='green')
ax2.bar(x + width/2, no_cache_latency, width, label='No Cache', alpha=0.8, color='red')
ax2.set_ylabel('Latency (seconds)')
ax2.set_title('Total Latency Comparison: Cache vs No Cache')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_latency, no_cache_latency)):
    ax2.text(i - width/2, cached + max(cached_latency) * 0.01, f'{cached:.2f}s', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, no_cache + max(no_cache_latency) * 0.01, f'{no_cache:.2f}s', ha='center', va='bottom', fontsize=9)

# Input tokens comparison
cached_input = [metrics[model]['cached']['input_tokens'] for model in models]
no_cache_input = [metrics[model]['no_cache']['input_tokens'] for model in models]

ax3.bar(x - width/2, cached_input, width, label='Cached', alpha=0.8, color='green')
ax3.bar(x + width/2, no_cache_input, width, label='No Cache', alpha=0.8, color='red')
ax3.set_ylabel('Input Tokens')
ax3.set_title('Total Input Tokens: Cache vs No Cache')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_input, no_cache_input)):
    ax3.text(i - width/2, cached + max(cached_input) * 0.01, f'{cached:,}', ha='center', va='bottom', fontsize=9)
    ax3.text(i + width/2, no_cache + max(no_cache_input) * 0.01, f'{no_cache:,}', ha='center', va='bottom', fontsize=9)

# Output tokens comparison
cached_output = [metrics[model]['cached']['output_tokens'] for model in models]
no_cache_output = [metrics[model]['no_cache']['output_tokens'] for model in models]

ax4.bar(x - width/2, cached_output, width, label='Cached', alpha=0.8, color='green')
ax4.bar(x + width/2, no_cache_output, width, label='No Cache', alpha=0.8, color='red')
ax4.set_ylabel('Output Tokens')
ax4.set_title('Total Output Tokens: Cache vs No Cache')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_output, no_cache_output)):
    ax4.text(i - width/2, cached + max(cached_output) * 0.01, f'{cached:,}', ha='center', va='bottom', fontsize=9)
    ax4.text(i + width/2, no_cache + max(no_cache_output) * 0.01, f'{no_cache:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('cache_vs_no_cache_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create savings/improvement chart
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

categories = ['Cost Savings (%)', 'Latency Improvement (%)', 'Input Token Reduction (%)']
micro_improvements = []
lite_improvements = []

for model in models:
    cached = metrics[model]['cached']
    no_cache = metrics[model]['no_cache']
    
    cost_savings = ((no_cache['cost'] - cached['cost']) / no_cache['cost']) * 100
    latency_improvement = ((no_cache['latency'] - cached['latency']) / no_cache['latency']) * 100
    input_reduction = ((no_cache['input_tokens'] - cached['input_tokens']) / no_cache['input_tokens']) * 100
    
    if 'micro' in model:
        micro_improvements = [cost_savings, latency_improvement, input_reduction]
    else:
        lite_improvements = [cost_savings, latency_improvement, input_reduction]

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, micro_improvements, width, label='Nova Micro v1', alpha=0.8)
ax.bar(x + width/2, lite_improvements, width, label='Nova Lite v1', alpha=0.8)

ax.set_ylabel('Improvement (%)')
ax.set_title('Cache Benefits: Percentage Improvements')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add value labels
for i, (micro, lite) in enumerate(zip(micro_improvements, lite_improvements)):
    ax.text(i - width/2, micro + 1, f'{micro:.1f}%', ha='center', va='bottom')
    ax.text(i + width/2, lite + 1, f'{lite:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('cache_improvements.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== CACHE VS NO-CACHE COMPARISON SUMMARY ===\n")

for model in models:
    print(f"{model.upper().replace('-', ' ')}:")
    cached = metrics[model]['cached']
    no_cache = metrics[model]['no_cache']
    
    cost_savings = ((no_cache['cost'] - cached['cost']) / no_cache['cost']) * 100
    latency_improvement = ((no_cache['latency'] - cached['latency']) / no_cache['latency']) * 100
    input_reduction = ((no_cache['input_tokens'] - cached['input_tokens']) / no_cache['input_tokens']) * 100
    
    print(f"  Requests processed: {cached['num_requests']}")
    print(f"  Cost: ${cached['cost']:.5f} (cached) vs ${no_cache['cost']:.5f} (no cache)")
    print(f"  → Cost savings: {cost_savings:.1f}%")
    print(f"  Latency: {cached['latency']:.2f}s (cached) vs {no_cache['latency']:.2f}s (no cache)")
    print(f"  → Latency improvement: {latency_improvement:.1f}%")
    print(f"  Input tokens: {cached['input_tokens']:,} (cached) vs {no_cache['input_tokens']:,} (no cache)")
    print(f"  → Input token reduction: {input_reduction:.1f}%")
    print(f"  Output tokens: {cached['output_tokens']:,} (cached) vs {no_cache['output_tokens']:,} (no cache)")
    print(f"  → Output token difference: {((cached['output_tokens'] - no_cache['output_tokens']) / no_cache['output_tokens']) * 100:.1f}%")
    print()

print("=== KEY INSIGHTS ===")
print("• Caching provides significant cost savings by reducing input token usage")
print("• Latency improvements from caching reduce overall processing time")
print("• Input token reduction is the primary driver of cost savings")
print("• Output tokens remain relatively consistent between cached and non-cached runs")
