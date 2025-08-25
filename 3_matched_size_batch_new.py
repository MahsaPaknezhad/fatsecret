import pandas as pd
import json
import time
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field
from typing import List, Optional

# Structured output models
class EatenInfo(BaseModel):
    singular_description: str = Field(description="Singular form of serving")
    plural_description: str = Field(description="Plural form of serving")
    units: float = Field(description="Number of units")
    metric_description: str = Field(description="ml or g")
    per_unit_metric_amount: float = Field(description="Metric amount per unit")
    total_metric_amount: float = Field(description="Total metric amount")
    imperial_description: str = Field(description="oz or fl oz")
    per_unit_imperial_amount: float = Field(description="Imperial amount per unit")
    total_imperial_amount: float = Field(description="Total imperial amount")

class SuggestedServing(BaseModel):
    serving_id: str = Field(description="ID of matched serving")
    serving: str = Field(description="Serving description name")
    is_default: int = Field(description="Default serving flag")
    serving_description: str = Field(description="Serving description")
    number_of_units: float = Field(description="Number of units for serving")

class Ingredient(BaseModel):
    food_id: int = Field(description="Selected food ID")
    food_name: str = Field(description="Selected food name")
    food_type: str = Field(description="Selected food type")
    brand_name: Optional[str] = Field(description="Brand name or empty")
    match_accuracy: int = Field(description="Match accuracy 1-100")
    eaten: EatenInfo
    suggested_serving: SuggestedServing

class FoodMatchResponse(BaseModel):
    query: List[str] = Field(description="List of original food queries")
    ingredients: List[Ingredient]

# Read system prompt
with open('/home/ubuntu/projects/fatsecret/prompts/prompt3.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip().strip('"')

# Read test data
df = pd.read_csv('/home/ubuntu/projects/fatsecret/data/test_data_clean.csv')
test_data = df[['Prompt 3 - match sizes Input', 'Prompt 3 - match sizes Output']].to_dict('records')

# Initialize model with structured output
model = ChatBedrock(
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.9
    }
).with_structured_output(FoodMatchResponse)

def process_food_item(food_data):
    try:
        foods = food_data.get('foods', [])
        all_results = []
        
        for food_item in foods:
            single_food_input = {
                'foods': [food_item],
                'input': food_data.get('input', ''),
                'language': food_data.get('language', ''),
                'language_description': food_data.get('language_description', ''),
                'region': food_data.get('region', ''),
                'region_description': food_data.get('region_description', '')
            }
            
            formatted_prompt = system_prompt.replace("{{foods}}", json.dumps(single_food_input, indent=2))
            
            start_time = time.time()
            response = model.invoke(formatted_prompt)
            invocation_time = time.time() - start_time
            
            result = {
                "food_item": food_item,
                "response": response.model_dump(),
                "invocation_time": invocation_time,
                "status": "success"
            }
            
            all_results.append(result)
            
        return all_results
        
    except Exception as e:
        return [{
            "food_item": None,
            "response": f"ERROR: {str(e)}",
            "invocation_time": None,
            "status": "error"
        }]

# Process all test data
all_results = []

for row_idx, test_case in enumerate(test_data):
    print(f"Processing row {row_idx + 1}/{len(test_data)}")
    
    input_data = json.loads(test_case['Prompt 3 - match sizes Input'])
    results = process_food_item(input_data)
    
    row_summary = {
        "row_index": row_idx,
        "food_count": len(input_data.get('foods', [])),
        "results": results,
        "total_time": sum(r.get('invocation_time', 0) or 0 for r in results)
    }
    
    all_results.append(row_summary)
    print(f"Row {row_idx + 1} completed with {len(results)} food items processed")
    
    if row_idx < len(test_data) - 1:
        time.sleep(1)

# Save results
with open('/home/ubuntu/projects/fatsecret/outputs/round3/3_matched_size_batch_new_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Completed processing {len(test_data)} rows with structured output")
