import pandas as pd
import json
import boto3
import asyncio
import time
import itertools
from concurrent.futures import ThreadPoolExecutor

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
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0"
]

model_cycle = itertools.cycle(models)
semaphore = asyncio.Semaphore(10)

async def invoke_food_async(food_item, user_message, model_id, executor):
    async with semaphore:
        print(f"{food_item['query']} - {model_id}")
        try:
            start_time = time.time()
            
            # Run the synchronous boto3 call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": system_prompt.replace("{{foods}}", user_message)}]}],
                    inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9}
                )
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

async def process_row(row_idx, test_case, executor):
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
        model_id = next(model_cycle)
        tasks.append(invoke_food_async(food_item, user_message, model_id, executor))
    
    # Process all foods in this row concurrently
    row_start_time = time.time()
    row_results = await asyncio.gather(*tasks)
    row_total_time = time.time() - row_start_time
    
    # Combine results for this row
    combined_responses = " ".join([result["actual"] for result in row_results])
    total_cost = sum(result["cost"] for result in row_results if result["cost"])
    
    return {
        "row_index": row_idx,
        "total_time": row_total_time,
        "food_count": len(input_data['foods']),
        "individual_results": row_results,
        "ingredients": combined_responses,
        "total_cost": total_cost
    }

async def main():
    all_results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for row_idx, test_case in enumerate(test_data):
            result = await process_row(row_idx, test_case, executor)
            all_results.append(result)
            print(f"Row {row_idx + 1} completed in {result['total_time']:.2f}s with {result['food_count']} foods")
    
    with open('outputs3/3_match_sizes_multi_model_asyncio.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Completed all {len(test_data)} rows")

if __name__ == "__main__":
    asyncio.run(main())
