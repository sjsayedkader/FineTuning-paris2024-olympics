import boto3
import json
import time


bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

resp = bedrock_runtime.invoke_model(
    modelId="arn:aws:bedrock:...:custom-model/llama3-8b-paris2024",
    body=json.dumps({
        "messages": [
            {"role": "user", "content": "How many medals did France win at Paris 2024?"}
        ]
    })
)

print(json.loads(resp["body"].read()))
