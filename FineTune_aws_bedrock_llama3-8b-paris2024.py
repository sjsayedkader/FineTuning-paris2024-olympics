import boto3
import json
import time

bedrock = boto3.client("bedrock", region_name="us-east-1")

job_name = "llama3-8b-paris2024-ft"

response = bedrock.create_model_customization_job(
    jobName=job_name,
    customModelName="llama3-8b-paris2024",
    roleArn="arn:aws:iam::926643152554:role/BedrockLlama3FineTuneRole",
    baseModelIdentifier="meta.llama3-8b-instruct-v1:0",
    trainingDataConfig={
        "s3Uri": "s3://bedrockfinetunesk/ft/paris2024/data/train.jsonl"
    },
    validationDataConfig={
        "s3Uri": "s3://bedrockfinetunesk/ft/paris2024/data/val.jsonl"
    },
    hyperParameters={
        "epochCount": "2",
        "batchSize": "2",
        "learningRate": "2e-5",
        "lrScheduler": "linear"
    },
    outputDataConfig={
        "s3Uri": "s3://<your-bucket>/ft/paris2024/output/"
    }
)

print("Started job:", response["jobArn"])
