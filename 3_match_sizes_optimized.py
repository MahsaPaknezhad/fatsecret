import pandas as pd
import json
import boto3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize client once
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Read and cache system prompt
with open('/home/ubuntu/projects/fatsecret/prompts/prompt3.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

def invoke_batch_optimized(input_data, row_idx):
    try:
        start_time = time.time()
        user_message = json.dumps(input_data, separators=(',', ':'))  # Compact JSON
        
        response = client.converse(
            modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
            messages=[{"role": "user", "content": [{"text": system_prompt.replace("{{foods}}", user_message)}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0}  # Reduced tokens, deterministic
        )
        
        invocation_time = time.time() - start_time
        response_text = response["output"]["message"]["content"][0]["text"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        cost = (input_tokens * 0.00024 / 1000) + (output_tokens * 0.00097 / 1000)
        
        return {
            "row_index": row_idx,
            "food_count": len(input_data['foods']),
            "invocation_time": invocation_time,
            "ingredients": response_text,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    except Exception as e:
        return {
            "row_index": row_idx,
            "food_count": len(input_data.get('foods', [])),
            "invocation_time": None,
            "ingredients": f"ERROR: {str(e)}",
            "cost": None,
            "input_tokens": None,
            "output_tokens": None
        }

# Process all rows in parallel
start_time = time.time()
tasks = []
for row_idx, test_case in enumerate(test_data):
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    tasks.append((input_data, row_idx))

all_results = []
with ThreadPoolExecutor(max_workers=5) as executor:  # Conservative concurrency
    futures = [executor.submit(invoke_batch_optimized, *task) for task in tasks]
    for future in as_completed(futures):
        result = future.result()
        all_results.append(result)
        print(f"Row {result['row_index'] + 1} completed in {result['invocation_time']:.2f}s" if result['invocation_time'] else f"Row {result['row_index'] + 1} failed")

# Sort results by row index
all_results.sort(key=lambda x: x['row_index'])

total_time = time.time() - start_time
print(f"Completed all {len(test_data)} rows in {total_time:.2f}s")

with open('/home/ubuntu/projects/fatsecret/outputs/round3/3_match_sizes_optimized.json', 'w') as f:
    json.dump(all_results, f, indent=2)
