import pandas as pd
import json
import boto3
import asyncio
import time
from botocore.exceptions import ClientError

# Read system prompt
with open('prompts/prompt3_old.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

# Create semaphore for rate limiting
semaphore = asyncio.Semaphore(3)

async def invoke_food_async(food_item, user_message, client):
    async with semaphore:
        print(f"Processing: {food_item['query']}")
        
        try:
            # Run the synchronous boto3 call in executor
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            response = await loop.run_in_executor(
                None,
                lambda: client.converse(
                    modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
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
                "actual": response_text,
                "invocation_time": invocation_time,
                "cost": cost
            }
            
        except Exception as e:
            return {
                "food_query": food_item['query'],
                "actual": f"ERROR: {str(e)}",
                "invocation_time": None,
                "cost": None
            }

async def process_row(test_case, row_idx, client):
    print(f"Processing row {row_idx + 1}/{len(test_data)}")
    
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    
    # Create tasks for all foods in this row
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
        tasks.append(invoke_food_async(food_item, user_message, client))
    
    # Execute all tasks for this row concurrently
    row_start_time = time.time()
    row_results = await asyncio.gather(*tasks)
    row_total_time = time.time() - row_start_time
    
    # Combine results
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
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    
    # Process all rows
    tasks = [process_row(test_case, idx, client) for idx, test_case in enumerate(test_data)]
    all_results = await asyncio.gather(*tasks)
    
    # Save results
    with open('outputs3/3_match_sizes_asyncio.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Completed all {len(test_data)} rows")

if __name__ == "__main__":
    asyncio.run(main())
