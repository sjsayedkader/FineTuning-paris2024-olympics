import boto3
import json
import time

bedrock = boto3.client("bedrock", region_name="us-east-1")

job_name = "llama3-8b-paris2024-ft"
while True:
    r = bedrock.get_model_customization_job(jobName=job_name)
    status = r["status"]
    print("Status:", status)
    if status in ["Failed", "Completed", "Stopped"]:
        break
    time.sleep(30)
