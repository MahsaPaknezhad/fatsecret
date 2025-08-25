import json
import os
import pandas as pd
from statistics import mean, stdev

def load_and_analyze():
    folder_path = 'outputs/round4'
    experiments = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            parts = filename.replace('.json', '').split('_')
            exp_name = '_'.join(parts[:-1]).replace('3_match_sizes_batch_', '')
            run_num = int(parts[-1])
            
            if exp_name not in experiments:
                experiments[exp_name] = {}
            
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                
            total_cost = sum(item.get('cost', 0) or 0 for item in data)
            total_latency = sum(item.get('invocation_time', 0) or 0 for item in data)
            total_input_tokens = sum(item.get('input_tokens', 0) or 0 for item in data)
            total_output_tokens = sum(item.get('output_tokens', 0) or 0 for item in data)
            
            experiments[exp_name][run_num] = {
                'cost': total_cost,
                'latency': total_latency,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'requests': len(data)
            }
    
    # Create detailed comparison
    results = []
    for exp_name, runs in experiments.items():
        for run_num, metrics in runs.items():
            results.append({
                'experiment': exp_name,
                'run': run_num,
                'cost': metrics['cost'],
                'latency': metrics['latency'],
                'input_tokens': metrics['input_tokens'],
                'output_tokens': metrics['output_tokens'],
                'total_tokens': metrics['input_tokens'] + metrics['output_tokens'],
                'cost_per_token': metrics['cost'] / (metrics['input_tokens'] + metrics['output_tokens']) if (metrics['input_tokens'] + metrics['output_tokens']) > 0 else 0,
                'tokens_per_second': (metrics['input_tokens'] + metrics['output_tokens']) / metrics['latency'] if metrics['latency'] > 0 else 0
            })
    
    df = pd.DataFrame(results)
    
    print("=== DETAILED COMPARISON BY RUN ===")
    print(df.round(8).to_string(index=False))
    
    print("\n=== SUMMARY STATISTICS ===")
    summary = df.groupby('experiment').agg({
        'cost': ['mean', 'std'],
        'latency': ['mean', 'std'],
        'input_tokens': ['mean', 'std'],
        'output_tokens': ['mean', 'std'],
        'total_tokens': ['mean', 'std'],
        'cost_per_token': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std']
    }).round(6)
    
    print(summary)
    
    print("\n=== RANKINGS ===")
    rankings = df.groupby('experiment').mean().round(6)
    
    print("\nBy Cost (lowest to highest):")
    cost_ranking = rankings.sort_values('cost')
    for i, (exp, row) in enumerate(cost_ranking.iterrows(), 1):
        print(f"{i}. {exp}: ${row['cost']:.6f}")
    
    print("\nBy Latency (fastest to slowest):")
    latency_ranking = rankings.sort_values('latency')
    for i, (exp, row) in enumerate(latency_ranking.iterrows(), 1):
        print(f"{i}. {exp}: {row['latency']:.2f}s")
    
    print("\nBy Cost per Token (most efficient to least):")
    efficiency_ranking = rankings.sort_values('cost_per_token')
    for i, (exp, row) in enumerate(efficiency_ranking.iterrows(), 1):
        print(f"{i}. {exp}: ${row['cost_per_token']:.2e} per token")
    
    print("\nBy Tokens per Second (fastest processing to slowest):")
    throughput_ranking = rankings.sort_values('tokens_per_second', ascending=False)
    for i, (exp, row) in enumerate(throughput_ranking.iterrows(), 1):
        print(f"{i}. {exp}: {row['tokens_per_second']:.2f} tokens/sec")

if __name__ == "__main__":
    load_and_analyze()
