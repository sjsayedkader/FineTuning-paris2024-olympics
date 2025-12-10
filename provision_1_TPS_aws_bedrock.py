import boto3
import json
import time

bedrock = boto3.client("bedrock", region_name="us-east-1")

job_name = "llama3-8b-paris2024-ft"
response = bedrock.create_provisioned_model_throughput(
    modelUnits=1,
    provisionedModelName="llama3-8b-paris2024-throughput",
    modelId="arn:aws:bedrock:...:custom-model/llama3-8b-paris2024"  # custom model arn
)

print(response)
