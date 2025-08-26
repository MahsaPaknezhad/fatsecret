import pandas as pd
import json
import boto3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import itertools

# Read system prompt
with open('prompts/prompt3_old.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Available models to cycle through
models = [
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-pro-v1:0"
]

# Create a cycle iterator for models
model_cycle = itertools.cycle(models)

# Rate limiting semaphore
# rate_limiter = Semaphore(10)

def invoke_food(food_item, user_message, model_id):
    #with rate_limiter:
    print(f"{food_item['query']} - {model_id}")
    try:
        start_time = time.time()
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": system_prompt.replace("{{foods}}",user_message)}]}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9}
        )
        invocation_time = time.time() - start_time
        
        response_text = response["output"]["message"]["content"][0]["text"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        cost = (input_tokens * 0.00024 / 1000) + (output_tokens * 0.00097 / 1000)
        
        return {
            "food_query": food_item['query'],
            "model_id": model_id,
            "actual": response_text,
            "invocation_time": invocation_time,
            "cost": cost
        }
    except Exception as e:
        return {
            "food_query": food_item['query'],
            "model_id": model_id,
            "actual": f"ERROR: {str(e)}",
            "invocation_time": None,
            "cost": None
        }

all_results = []

for row_idx, test_case in enumerate(test_data):
    print(f"Processing row {row_idx + 1}/{len(test_data)}")
    
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    
    # Prepare tasks for this row with different models
    tasks = []
    for food_item in input_data['foods']:
        single_food_input = {
            "input": input_data["input"],
            "language": input_data["language"],
            "region": input_data["region"],
            "language_description": input_data["language_description"],
            "region_description": input_data["region_description"],
            "include_servings": input_data["include_servings"],
            "foods": [food_item]
        }
        user_message = json.dumps(single_food_input, indent=2)
        model_id = next(model_cycle)  # Get next model in cycle
        tasks.append((food_item, user_message, model_id))
    
    # Process with limited concurrency
    row_start_time = time.time()
    row_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(invoke_food, *task) for task in tasks]
        for future in as_completed(futures):
            row_results.append(future.result())
    
    row_total_time = time.time() - row_start_time
    
    # Combine results for this row
    combined_responses = " ".join([result["actual"] for result in row_results])
    total_cost = sum(result["cost"] for result in row_results if result["cost"])
    
    row_summary = {
        "row_index": row_idx,
        "total_time": row_total_time,
        "food_count": len(input_data['foods']),
        "individual_results": row_results,
        "ingredients": combined_responses,
        "total_cost": total_cost
    }
    
    all_results.append(row_summary)
    print(f"Row {row_idx + 1} completed in {row_total_time:.2f}s with {len(input_data['foods'])} foods")

with open('outputs3/3_match_sizes_multi_model.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Completed all {len(test_data)} rows")
