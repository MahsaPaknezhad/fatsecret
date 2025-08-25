import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
files = {
    'nova-micro-v1_cached': 'outputs/1_extract_foods_nova-micro-v2_cached.json',
    'nova-micro-v1_no_cache': 'outputs/1_extract_foods_nova-micro-v2_no_cache.json',
    'nova-lite-v1_cached': 'outputs/1_extract_foods_nova-lite-v1_cached.json',
    'nova-lite-v1_no_cache': 'outputs/1_extract_foods_nova-lite-v1_no_cache.json'
}

data = {}
for key, file_path in files.items():
    with open(file_path, 'r') as f:
        data[key] = json.load(f)

# Extract metrics
metrics = {}
for key, content in data.items():
    model = key.split('_')[0]
    cache_status = 'cached' if 'cached' in key else 'no_cache'
    
    if model not in metrics:
        metrics[model] = {}
    
    metrics[model][cache_status] = {
        'cost': content['total_cost'],
        'latency': content['total_latency'],
        'input_tokens': content['total_input_tokens'],
        'output_tokens': content['total_output_tokens']
    }

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

models = list(metrics.keys())
x = np.arange(len(models))
width = 0.35

# Cost comparison
cached_costs = [metrics[model]['cached']['cost'] for model in models]
no_cache_costs = [metrics[model]['no_cache']['cost'] for model in models]

ax1.bar(x - width/2, cached_costs, width, label='Cached', alpha=0.8)
ax1.bar(x + width/2, no_cache_costs, width, label='No Cache', alpha=0.8)
ax1.set_ylabel('Cost ($)')
ax1.set_title('Cost Comparison: Cache vs No Cache')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_costs, no_cache_costs)):
    ax1.text(i - width/2, cached + 0.0001, f'${cached:.4f}', ha='center', va='bottom')
    ax1.text(i + width/2, no_cache + 0.0001, f'${no_cache:.4f}', ha='center', va='bottom')

# Latency comparison
cached_latency = [metrics[model]['cached']['latency'] for model in models]
no_cache_latency = [metrics[model]['no_cache']['latency'] for model in models]

ax2.bar(x - width/2, cached_latency, width, label='Cached', alpha=0.8)
ax2.bar(x + width/2, no_cache_latency, width, label='No Cache', alpha=0.8)
ax2.set_ylabel('Latency (seconds)')
ax2.set_title('Latency Comparison: Cache vs No Cache')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_latency, no_cache_latency)):
    ax2.text(i - width/2, cached + 0.1, f'{cached:.1f}s', ha='center', va='bottom')
    ax2.text(i + width/2, no_cache + 0.1, f'{no_cache:.1f}s', ha='center', va='bottom')

# Input tokens comparison
cached_input = [metrics[model]['cached']['input_tokens'] for model in models]
no_cache_input = [metrics[model]['no_cache']['input_tokens'] for model in models]

ax3.bar(x - width/2, cached_input, width, label='Cached', alpha=0.8)
ax3.bar(x + width/2, no_cache_input, width, label='No Cache', alpha=0.8)
ax3.set_ylabel('Input Tokens')
ax3.set_title('Input Tokens: Cache vs No Cache')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_input, no_cache_input)):
    ax3.text(i - width/2, cached + 50, f'{cached:,}', ha='center', va='bottom')
    ax3.text(i + width/2, no_cache + 50, f'{no_cache:,}', ha='center', va='bottom')

# Output tokens comparison
cached_output = [metrics[model]['cached']['output_tokens'] for model in models]
no_cache_output = [metrics[model]['no_cache']['output_tokens'] for model in models]

ax4.bar(x - width/2, cached_output, width, label='Cached', alpha=0.8)
ax4.bar(x + width/2, no_cache_output, width, label='No Cache', alpha=0.8)
ax4.set_ylabel('Output Tokens')
ax4.set_title('Output Tokens: Cache vs No Cache')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()

# Add value labels on bars
for i, (cached, no_cache) in enumerate(zip(cached_output, no_cache_output)):
    ax4.text(i - width/2, cached + 10, f'{cached:,}', ha='center', va='bottom')
    ax4.text(i + width/2, no_cache + 10, f'{no_cache:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('cache_vs_no_cache_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== CACHE VS NO-CACHE COMPARISON SUMMARY ===\n")

for model in models:
    print(f"{model.upper()}:")
    cached = metrics[model]['cached']
    no_cache = metrics[model]['no_cache']
    
    cost_savings = ((no_cache['cost'] - cached['cost']) / no_cache['cost']) * 100
    latency_improvement = ((no_cache['latency'] - cached['latency']) / no_cache['latency']) * 100
    input_reduction = ((no_cache['input_tokens'] - cached['input_tokens']) / no_cache['input_tokens']) * 100
    
    print(f"  Cost: ${cached['cost']:.4f} (cached) vs ${no_cache['cost']:.4f} (no cache)")
    print(f"  → Cost savings: {cost_savings:.1f}%")
    print(f"  Latency: {cached['latency']:.1f}s (cached) vs {no_cache['latency']:.1f}s (no cache)")
    print(f"  → Latency improvement: {latency_improvement:.1f}%")
    print(f"  Input tokens: {cached['input_tokens']:,} (cached) vs {no_cache['input_tokens']:,} (no cache)")
    print(f"  → Input token reduction: {input_reduction:.1f}%")
    print(f"  Output tokens: {cached['output_tokens']:,} (cached) vs {no_cache['output_tokens']:,} (no cache)")
    print()
