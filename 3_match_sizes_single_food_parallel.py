import pandas as pd
import json
import boto3
import time
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Read system prompt
with open('prompts/prompt3.txt', 'r', encoding='utf-8') as f:
    prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

# Read models configuration
models_df = pd.read_csv('data/models/step3_2.csv')

def invoke_model(model_row, user_message, food_query, expected_output):
    if "(latency_optimized)" in model_row['model']:
        model_id = model_row['model'].replace("(latency_optimized)", "").strip()
        performance = "optimized"
    else:
        model_id = model_row['model'].strip()
        performance = "standard"
    
    region = model_row['region'].strip()
    input_price = float(model_row['input price'].replace('$', ''))
    output_price = float(model_row['output price'].replace('$', ''))
    
    print(f"Testing model: {model_id} with food query: {food_query}")
    
    # Create client for this region
    client = boto3.client("bedrock-runtime", region_name=region)
    
    print(prompt.replace("{{foods}}",user_message))
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt.replace("{{foods}}",user_message)}],
        }
    ]
    
    try:
        # Retry mechanism
        for attempt in range(3):
            try:
                start_time = time.time()
                response = client.converse(
                    modelId=model_id,
                    messages=conversation,
                    inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9},
                    performanceConfig={"latency": performance}
                )
                end_time = time.time()
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        invocation_time = end_time - start_time
        
        # Handle different response structures based on model type
        if "gpt" in model_id.lower():
            content = response["output"]["message"]["content"][0]
            if "text" in content:
                response_text = content["text"]
            else:
                response_text = str(content)
        else:
            response_text = response["output"]["message"]["content"][0]["text"]
        
        # Calculate cost
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        cost = (input_tokens * input_price / 1000) + (output_tokens * output_price / 1000)
        
        result = {
            "model": model_row['model'].strip(),
            "region": region,
            "food_query": food_query,
            "input": user_message,
            "expected": expected_output,
            "actual": response_text,
            "invocation_time": invocation_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "success": True
        }
        
    except (ClientError, Exception) as e:
        result = {
            "model": model_id,
            "region": region,
            "food_query": food_query,
            "input": user_message,
            "expected": expected_output,
            "actual": f"ERROR: {str(e)}",
            "invocation_time": None,
            "input_tokens": None,
            "output_tokens": None,
            "cost": None,
            "success": False
        }
    
    time_str = f"{result.get('invocation_time'):.2f}" if result.get('invocation_time') else "N/A"
    print(f"Completed test for {model_id} with food {food_query} (took {time_str}s)")
    
    return result

# Prepare all tasks
tasks = []
for test_case in test_data:
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    expected_output = test_case['Prompt 3 - match sizes Output']
    
    # Process each food item individually
    for food_item in input_data['foods']:
        # Create new input with single food item
        single_food_input = {
            "input": input_data["input"],
            "language": input_data["language"],
            "region": input_data["region"],
            "language_description": input_data["language_description"],
            "region_description": input_data["region_description"],
            "include_servings": input_data["include_servings"],
            "foods": [food_item]  # Only one food item
        }
        
        user_message = json.dumps(single_food_input, indent=2)
        
        for _, model_row in models_df.iterrows():
            tasks.append((model_row, user_message, food_item['query'], expected_output))

# Execute tasks in parallel
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_task = {executor.submit(invoke_model, *task): task for task in tasks}
    
    for future in as_completed(future_to_task):
        result = future.result()
        results.append(result)

# Save results
with open('outputs/round2/3_match_sizes_single_food_parallel_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Completed testing {len(models_df)} models with individual food items in parallel")
print(f"Results saved to 3_match_sizes_single_food_parallel_results.json")
