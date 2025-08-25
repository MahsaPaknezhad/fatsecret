import pandas as pd
import json
import boto3
import time

# Read system prompt
with open('/home/ubuntu/projects/fatsecret/prompts/prompt3.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

def invoke_batch(input_data):
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    try:
        start_time = time.time()
        user_message = json.dumps(input_data, indent=2)
        
        response = client.converse(
            modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
            messages=[{"role": "user", "content": [{"text": system_prompt.replace("{{foods}}", user_message)}]}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9}
        )
        
        invocation_time = time.time() - start_time
        response_text = response["output"]["message"]["content"][0]["text"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        cost = (input_tokens * 0.00024 / 1000) + (output_tokens * 0.00097 / 1000)
        
        return {
            "actual": response_text,
            "invocation_time": invocation_time,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    except Exception as e:
        return {
            "actual": f"ERROR: {str(e)}",
            "invocation_time": None,
            "cost": None,
            "input_tokens": None,
            "output_tokens": None
        }

all_results = []

for row_idx, test_case in enumerate(test_data):
    print(f"Processing row {row_idx + 1}/{len(test_data)}")
    
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    result = invoke_batch(input_data)
    
    row_summary = {
        "row_index": row_idx,
        "food_count": len(input_data['foods']),
        "invocation_time": result["invocation_time"],
        "ingredients": result["actual"],
        "cost": result["cost"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"]
    }
    
    all_results.append(row_summary)
    print(f"Row {row_idx + 1} completed in {result['invocation_time']:.2f}s with {len(input_data['foods'])} foods")
    
    if row_idx < len(test_data) - 1:
        time.sleep(3)

with open('/home/ubuntu/projects/fatsecret/outputs/round3/3_match_sizes_batch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Completed all {len(test_data)} rows")
