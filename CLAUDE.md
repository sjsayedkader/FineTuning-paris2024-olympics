# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project demonstrates an end-to-end LLM fine-tuning pipeline using **Databricks**, **AWS Bedrock**, and **Meta Llama 3 8B Instruct**. The goal is to teach a foundation LLM updated knowledge about the Paris 2024 Olympics through supervised fine-tuning.

### Key Components

1. **Data Source**: Paris 2024 Olympics dataset (Kaggle CSV files in `paris-dataset/`)
2. **Data Platform**: Databricks Delta tables for structured Olympic data
3. **Training Data Generation**: LLM-powered Q&A pair generation using Databricks `ai_query()` with Llama 3
4. **Fine-Tuning**: AWS Bedrock model customization jobs
5. **Deployment**: Custom Bedrock model with provisioned throughput

## Architecture Flow

```
Kaggle CSVs → Databricks Delta Tables → ai_query() Q&A Generation →
JSONL Dataset → S3 → Bedrock Fine-Tuning → Custom Model Endpoint
```

## Key Scripts

### AWS Bedrock Fine-Tuning Workflow

1. **FineTune_aws_bedrock_llama3-8b-paris2024.py**: Initiates a Bedrock model customization job
   - Base model: `meta.llama3-8b-instruct-v1:0`
   - Training data: `s3://bedrockfinetunesk/ft/paris2024/data/train.jsonl`
   - Hyperparameters: 2 epochs, batch size 2, learning rate 2e-5

2. **FineTune_poll_job.py**: Polls Bedrock job status until completion/failure

3. **provision_1_TPS_aws_bedrock.py**: Provisions 1 TPS (tokens per second) throughput for the custom model

4. **infrence_custome_bedrock_model_paris2024.py**: Invokes the fine-tuned model for inference

### Databricks Scripts

- **MLFlow_gen-ai.py**: Example of using Databricks serving endpoints via OpenAI-compatible client with MLflow tracking

### Notebooks

- **Build your first AI agent.ipynb**: Complete guide to building a Databricks agent using Mosaic AI Agent Framework
  - Uses Foundation Model API (Claude or Llama 3.3)
  - Integrates `system.ai.python_exec` tool for code execution
  - Demonstrates MLflow ResponsesAgent pattern for logging and deployment

- **Convert delta table to jsonl.ipynb**: Converts Delta tables to JSONL format
- **convert delta table to jsonl chat.ipynb**: Converts to chat-formatted JSONL

## Data Structure

The `paris-dataset/` directory contains:
- **athletes.csv**: Athlete profiles (name, country, disciplines, height, weight, birth info)
- **medallists.csv**: Medal winners
- **medals.csv**, **medals_total.csv**: Medal counts
- **events.csv**: Competition events
- **schedules.csv**: Event schedules
- **teams.csv**: Team information
- **venues.csv**: Competition venues

## SQL Queries (`queries/` directory)

Training data generation queries using Databricks `ai_query()`:

- **Create QA Pair for athlets.sql**: Generates Q&A pairs from athlete data using Llama 3.1 8B
  - Samples 8 random athletes per batch
  - Prompts LLM to generate 6 diverse Q&A pairs
  - Parses JSON Lines output and inserts into `olympics_ft_training` table

- **create medalist batch.sql**: Q&A generation for medal winners
- **Create events batch.sql**: Q&A generation for events
- **CreACate schedules batch.sql**: Q&A generation for schedules
- **Create teams batch.sql**: Q&A generation for teams
- **Create_medal_batch.sql**: Q&A generation for medal counts
- **athlete_batch_temp table.sql**: Temporary table for athlete batches
- **split train validation data.sql**: Splits data into train/validation sets
- **Create training data table.sql**: Creates the training data table schema

## Training Data Format

Bedrock expects JSONL format with chat messages:

```json
{"messages":[
  {"role":"user","content":"Who won the Women's 400m Gold medal?"},
  {"role":"assistant","content":"PAULINO Marileidy won the Women's 400m Gold medal."}
]}
```

Training files:
- **paris2024_ft.jsonl**: Raw Q&A pairs (240 samples)
- **paris2024_ft_chat.jsonl**: Chat-formatted version

## AWS Resources

### IAM Configuration

- **BedrockLlama3FineTuneRole_policy.json**: IAM policy for S3 access and CloudWatch logging
- **BedrockLlama3FineTuneRole_trust_policy.json**: Trust relationship for Bedrock service

### S3 Bucket Structure

```
s3://bedrockfinetunesk/
  ft/paris2024/
    data/
      train.jsonl
      val.jsonl
    output/
```

## Python Environment

The project uses a virtual environment in `myenv/` with key dependencies:
- boto3 (AWS SDK)
- databricks-sdk
- databricks-openai
- databricks-agents
- mlflow
- Various data science libraries (pandas, numpy, etc.)

## Databricks Workspace Configuration

Required environment variables for Databricks operations:
```
DATABRICKS_TOKEN=<your-token>
DATABRICKS_HOST=https://dbc-XXXXX.cloud.databricks.com
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
```

## Common Workflows

### Fine-Tune a Model on Bedrock

```bash
# 1. Upload training data to S3
# 2. Start fine-tuning job
python FineTune_aws_bedrock_llama3-8b-paris2024.py

# 3. Monitor job status
python FineTune_poll_job.py

# 4. Provision throughput
python provision_1_TPS_aws_bedrock.py

# 5. Test inference
python infrence_custome_bedrock_model_paris2024.py
```

### Generate Training Data in Databricks

1. Load CSV files into Delta tables
2. Run SQL queries in `queries/` directory sequentially:
   - Create training data table
   - Run batch generation queries (athletes, medalists, events, schedules, teams, medals)
   - Split train/validation data
3. Export to JSONL format using conversion notebooks

## Important Notes

- The `myenv/` directory is a Python virtual environment and should be excluded from version control
- AWS region is set to `us-east-1` in all Bedrock scripts
- Databricks uses Unity Catalog for function discovery (`system.ai.python_exec`)
- The fine-tuning uses LoRA (Low-Rank Adaptation) approach via Bedrock
- Model ARNs and S3 bucket names are specific to the AWS account `926643152554`
