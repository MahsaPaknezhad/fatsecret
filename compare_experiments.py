import json
import statistics

def load_experiment_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_experiment(data, name):
    costs = [entry['cost'] for entry in data if entry.get('cost') is not None]
    latencies = [entry['invocation_time'] for entry in data if entry.get('invocation_time') is not None]
    
    return {
        'name': name,
        'total_requests': len(data),
        'total_cost': sum(costs),
        'avg_cost_per_request': statistics.mean(costs),
        'total_latency': sum(latencies),
        'avg_latency_per_request': statistics.mean(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'median_latency': statistics.median(latencies)
    }

# Load the three experiments
experiments = [
    ('Ultra Optimized', '/home/ubuntu/projects/fatsecret/outputs/round4/3_match_sizes_batch_ultra.json'),
    ('New Batch', '/home/ubuntu/projects/fatsecret/outputs/round4/3_match_sizes_batch_new.json'),
    ('Old Batch', '/home/ubuntu/projects/fatsecret/outputs/round4/3_match_sizes_batch_old.json')
]

results = []
for name, file_path in experiments:
    data = load_experiment_data(file_path)
    results.append(analyze_experiment(data, name))

# Print comparison
print("EXPERIMENT COMPARISON - COST AND LATENCY")
print("=" * 60)

print(f"{'Metric':<25} {'Ultra Optimized':<15} {'New Batch':<15} {'Old Batch':<15}")
print("-" * 75)

print(f"{'Total Requests':<25} {results[0]['total_requests']:<15} {results[1]['total_requests']:<15} {results[2]['total_requests']:<15}")
print(f"{'Total Cost ($)':<25} {results[0]['total_cost']:<15.6f} {results[1]['total_cost']:<15.6f} {results[2]['total_cost']:<15.6f}")
print(f"{'Avg Cost/Request ($)':<25} {results[0]['avg_cost_per_request']:<15.6f} {results[1]['avg_cost_per_request']:<15.6f} {results[2]['avg_cost_per_request']:<15.6f}")
print(f"{'Total Latency (s)':<25} {results[0]['total_latency']:<15.2f} {results[1]['total_latency']:<15.2f} {results[2]['total_latency']:<15.2f}")
print(f"{'Avg Latency/Request (s)':<25} {results[0]['avg_latency_per_request']:<15.2f} {results[1]['avg_latency_per_request']:<15.2f} {results[2]['avg_latency_per_request']:<15.2f}")
print(f"{'Min Latency (s)':<25} {results[0]['min_latency']:<15.2f} {results[1]['min_latency']:<15.2f} {results[2]['min_latency']:<15.2f}")
print(f"{'Max Latency (s)':<25} {results[0]['max_latency']:<15.2f} {results[1]['max_latency']:<15.2f} {results[2]['max_latency']:<15.2f}")
print(f"{'Median Latency (s)':<25} {results[0]['median_latency']:<15.2f} {results[1]['median_latency']:<15.2f} {results[2]['median_latency']:<15.2f}")

print("\nKEY INSIGHTS:")
print("-" * 30)

# Cost comparison
best_cost = min(results, key=lambda x: x['total_cost'])
worst_cost = max(results, key=lambda x: x['total_cost'])
cost_savings = ((worst_cost['total_cost'] - best_cost['total_cost']) / worst_cost['total_cost']) * 100

print(f"• Best cost performance: {best_cost['name']} (${best_cost['total_cost']:.6f})")
print(f"• Worst cost performance: {worst_cost['name']} (${worst_cost['total_cost']:.6f})")
print(f"• Cost savings: {cost_savings:.1f}% ({best_cost['name']} vs {worst_cost['name']})")

# Latency comparison
best_latency = min(results, key=lambda x: x['avg_latency_per_request'])
worst_latency = max(results, key=lambda x: x['avg_latency_per_request'])
latency_improvement = ((worst_latency['avg_latency_per_request'] - best_latency['avg_latency_per_request']) / worst_latency['avg_latency_per_request']) * 100

print(f"• Best latency performance: {best_latency['name']} ({best_latency['avg_latency_per_request']:.2f}s avg)")
print(f"• Worst latency performance: {worst_latency['name']} ({worst_latency['avg_latency_per_request']:.2f}s avg)")
print(f"• Latency improvement: {latency_improvement:.1f}% ({best_latency['name']} vs {worst_latency['name']})")
