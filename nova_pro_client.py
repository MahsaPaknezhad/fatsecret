import boto3

def call_nova_pro(prompt, region='us-east-2'):
    """Call Amazon Nova Pro using converse API with latency optimization"""
    client = boto3.client('bedrock-runtime', region_name=region)
    
    response = client.converse(
        modelId='us.amazon.nova-pro-v1:0:latency-optimized',
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={
            'maxTokens': 1000,
            'temperature': 0.1
        }
    )
    
    return response
