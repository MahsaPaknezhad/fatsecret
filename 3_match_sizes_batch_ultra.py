import pandas as pd
import json
import boto3
import time

# Read system prompt
with open('/home/ubuntu/projects/fatsecret/prompts/prompt3_ultra.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

def invoke_batch(input_data):
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    try:
        # Extract metadata
        metadata_keys = ['input', 'language', 'language_description', 'region', 'region_description', 'include_servings']
        metadata = {k: input_data[k] for k in metadata_keys if k in input_data}
        
        # Extract and store food metadata for restoration
        food_metadata = []
        optimized_foods = []
        
        for food in input_data['foods']:
            # Store removable data per food
            food_meta = {'query': food.get('query')}
            
            optimized_results = []
            for result in food['results']:
                # Store removable result data
                result_meta = {
                    'brand_name': result.get('brand_name', ''),
                    'food_type': result.get('food_type', '')
                }
                food_meta.setdefault('results_meta', []).append(result_meta)
                
                # Keep only essential data for model
                optimized_result = {
                    'food_id': result['food_id'],
                    'food_name': result['food_name'],
                    'servings': result['servings']
                }
                optimized_results.append(optimized_result)
            
            food_metadata.append(food_meta)
            optimized_foods.append({'results': optimized_results})
        
        # Create minimal model input
        model_input = {
            'input': input_data['input'],
            'foods': optimized_foods
        }
        
        user_message = json.dumps(model_input, indent=2)
        start_time = time.time()
        
        response = client.converse(
            modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
            messages=[{"role": "user", "content": [{"text": system_prompt.replace("{{foods}}", user_message)}]}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9}
        )
        
        invocation_time = time.time() - start_time
        response_text = response["output"]["message"]["content"][0]["text"].strip()
        
        # Clean JSON response
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0]
        
        response_json = json.loads(response_text)
        
        # Restore metadata to response
        response_json.update(metadata)
        
        # Restore food metadata to ingredients
        if 'ingredients' in response_json:
            for i, ingredient in enumerate(response_json['ingredients']):
                food_id = str(ingredient.get('food_id', ''))
                
                # Find matching food metadata by food_id
                for food_idx, food_meta in enumerate(food_metadata):
                    for result_idx, result_meta in enumerate(food_meta.get('results_meta', [])):
                        # Match by checking if this food_id exists in original data
                        original_food = input_data['foods'][food_idx]['results'][result_idx]
                        if str(original_food['food_id']) == food_id:
                            ingredient.update(result_meta)
                            break
        
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        cost = (input_tokens * 0.00024 + output_tokens * 0.00097) / 1000
        
        return {
            "actual": response_json,
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
    time_str = f"{result['invocation_time']:.2f}s" if result['invocation_time'] is not None else "ERROR"
    print(f"Row {row_idx + 1} completed in {time_str} with {len(input_data['foods'])} foods")
    
    if row_idx < len(test_data) - 1:
        time.sleep(3)

with open('/home/ubuntu/projects/fatsecret/outputs/round4/3_match_sizes_batch_ultra.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Completed all {len(test_data)} rows")
