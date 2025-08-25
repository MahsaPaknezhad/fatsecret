import pandas as pd
import json
import boto3
import time

# Read system prompt
with open('/home/ubuntu/projects/fatsecret/prompts/prompt1.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')


# Read test data
df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/test_data_clean.csv')
test_data = df[['Prompt 1 - Extract foods Input', 'Prompt 1 - Extract foods Output']].to_dict('records')

# Read models from step1.csv
models_df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/models/step1.csv')
models_df = models_df[models_df['model'].notna()]
models = models_df[['model', 'region', 'input price', 'input price (cache read)', 'output price']].to_dict('records')

def invoke_batch(input_data, model_id, region, use_cache=False, pricing=None):
    client = boto3.client("bedrock-runtime", region_name=region)
    #try:
    start_time = time.time()
    user_message = json.dumps(input_data, indent=2)
    
    if use_cache:
        messages = [
            {"role": "user", "content": [{"text": system_prompt}, {"cachePoint": {"type": "default"}}]},
            {"role": "user", "content": [{"text": user_message}]},
            {"role": "assistant","content": [{"text": " Here is the JSON response: ```json"}]}
        ]
    else:
        messages = [{"role": "user", "content": [{"text": system_prompt}]},
                        {"role": "user", "content": [{"text": user_message}]},
                    {"role": "assistant","content": [{"text": " Here is the JSON response: ```json"}]}]
    
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={
            "maxTokens": 2048, 
            "temperature": 0.1, 
            "topP": 0.9,
        }
    )
    
    invocation_time = time.time() - start_time
    response_text = response["output"]["message"]["content"][0]["text"]
    input_tokens = response["usage"]["inputTokens"]
    output_tokens = response["usage"]["outputTokens"]
    
    # Extract JSON from response text (remove markdown formatting)
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    json_content = response_text[json_start:json_end]
    response_json = json.loads(json_content)
    # Calculate cost using pricing from CSV
    input_price = float(pricing['input price (cache read)'].replace('$', '')) if use_cache else float(pricing['input price'].replace('$', ''))
    output_price = float(pricing['output price'].replace('$', ''))
    cost = (input_tokens * input_price / 1000) + (output_tokens * output_price / 1000)
    
    return {
        "actual": response_json,
        "invocation_time": invocation_time,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    '''except Exception as e:
        return e'''
    '''{
            "actual": f"ERROR: {str(e)}",
            "invocation_time": None,
            "cost": None,
            "input_tokens": None,
            "output_tokens": None
        }'''

for model_config in models:
    model_id = model_config['model']
    region = model_config['region']
    model_name = model_id.split('.')[-1].split(':')[0]
    
    for use_cache in [False, True]:
        cache_suffix = "_cached" if use_cache else "_no_cache"
        print(f"\nRunning {model_id} {'with' if use_cache else 'without'} caching")
        all_results = []

        for row_idx, test_case in enumerate(test_data):
            print(f"Processing row {row_idx + 1}/{len(test_data)}")
            
            input_data = json.loads(test_case['Prompt 1 - Extract foods Input'])
            result = invoke_batch(input_data, model_id, region, use_cache, model_config)
            
            row_summary = {
                "row_index": row_idx,
                "input_data": input_data,
                "invocation_time": result["invocation_time"],
                "extracted_foods": result["actual"],
                "cost": result["cost"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"]
            }
            
            all_results.append(row_summary)
            time_str = f"{result['invocation_time']:.2f}s" if result['invocation_time'] is not None else "ERROR"
            print(f"Row {row_idx + 1} completed in {time_str}")
            
            if row_idx < len(test_data) - 1:
                time.sleep(1)

        with open(f'/home/ubuntu/projects/fatsecret/outputs/1_extract_foods_{model_name}{cache_suffix}_1.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Completed {model_id}{cache_suffix}")
