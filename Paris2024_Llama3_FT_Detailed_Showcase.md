# Paris 2024 Olympics QA -- Llama 3 8B Fine-Tuning Showcase

## Overview

This project demonstrates an end‑to‑end Generative AI pipeline built
using **Databricks**, **AWS Bedrock**, and **Meta Llama 3 8B Instruct**.

The goal:\
**Teach a foundation LLM updated knowledge about the Paris 2024 Olympics
--- information beyond the base model's cutoff.**

This enables accurate question‑answering about:
- Medal winners and medal counts
- Athlete details (nationality, disciplines, physical stats, birth info)
- Event schedules and venues
- Team compositions and competitions
- Country performance metrics

### Project Highlights

- **Zero manual data labeling**: Q&A pairs auto-generated using LLM-to-LLM transformation
- **Production-ready pipeline**: Reproducible workflow from raw CSV to deployed API
- **Cost-optimized**: Uses serverless Databricks inference + lowest-tier Bedrock provisioning (1 TPS)
- **Enterprise architecture**: IAM roles, S3 data lakes, Unity Catalog integration

------------------------------------------------------------------------

## 1. Business Use Case

Organizations frequently need LLMs updated with post‑cutoff data: - new
products - regulations - financial changes - recent events (e.g., Paris
2024)

This project demonstrates how to: 1. Ingest fresh structured data\
2. Transform it into supervised Q&A\
3. Fine‑tune **Llama 3 8B**\
4. Deploy a private custom Bedrock model

------------------------------------------------------------------------

## 2. Architecture Diagram

                    ┌────────────────────────┐
                    │   Kaggle Olympics Data │
                    │ (CSV: athletes, etc.) │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     Databricks Delta   │
                    │  (Raw Olympic Tables)  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Databricks ai_query()  │
                    │ Llama 3 8B Serverless  │
                    │ Auto‑Generate Q&A Pairs│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   JSONL Chat Dataset   │
                    │ {messages:[{user..}]}  │
                    └───────────┬────────────┘
                                │ Upload
                                ▼
                    ┌────────────────────────┐
                    │          S3            │
                    │  train.jsonl (Q&A)     │
                    └───────────┬────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │   AWS Bedrock Fine‑Tuning Service  │
              │  Base: meta.llama3‑8b‑instruct     │
              │  Custom Model: paris2024-llama3    │
              └──────────────────┬─────────────────┘
                                 │ Provision 1 TPS
                                 ▼
                    ┌────────────────────────┐
                    │   Fine‑Tuned Endpoint  │
                    │  (bedrock-runtime)     │
                    └────────────────────────┘

------------------------------------------------------------------------

## 3. Data Engineering & Transformation

### Data Ingestion

**Source**: Kaggle Paris 2024 Olympics dataset (14 CSV files in `paris-dataset/`)

Loaded into Databricks Delta tables in catalog `workspace`, schema `sk_paris_olympics_2024`:

| Table | Key Fields | Records |
|-------|-----------|---------|
| **athletes** | name, country, disciplines, events, height, weight, birth_date, nationality | Athletes competing |
| **medallists** | name, medal_type, event, team, discipline, country, medal_date | Medal winners |
| **medals_total** | country, gold_medal, silver_medal, bronze_medal, total | Country medal counts |
| **events** | event, sport, sport_code, tag | Competition events |
| **schedules** | event, venue, date_start, date_end | Event timing |
| **teams** | team, event, country, team_gender | Team compositions |
| **venues** | venue_name, sports | Competition locations |

### LLM-Powered Transformation

#### The Innovation: Synthetic Q&A Generation

Instead of manual data labeling, this project uses **Llama 3.1 8B** (via Databricks `ai_query()`) to transform structured data into natural language Q&A pairs.

**Why this matters**:
- Eliminates weeks of manual annotation work
- Scales to any new dataset with similar SQL pattern
- Maintains factual accuracy by constraining LLM to provided data only

#### SQL Pattern for Q&A Generation

Each source table gets a dedicated query following this pattern:

```sql
-- File: queries/create medalist batch.sql
USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH sample AS (
  -- Step 1: Sample random rows (8 at a time to fit context window)
  SELECT medal_date, medal_type, name, gender, country, discipline, event
  FROM medallists
  WHERE is_medallist = true
  ORDER BY rand()
  LIMIT 8
),
medallists_batch AS (
  -- Step 2: Convert to JSON for LLM consumption
  SELECT to_json(collect_list(named_struct(
    'medal_date', medal_date,
    'medal_type', medal_type,
    'name', name,
    'gender', gender,
    'country', country,
    'discipline', discipline,
    'event', event
  ))) AS rows_json
  FROM sample
),
prompt AS (
  -- Step 3: Craft prompt with strict output format instructions
  SELECT
    'You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about Paris 2024 Olympic medallists.

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs.
- Mix question types:
  • "Who won the <medal_type> medal in <event>?"
  • "Which country does <name> represent?"
- Answers must be short factual sentences.
- Output STRICTLY in JSON Lines format:
  {"input": "<question>", "output": "<answer>"}
- No extra explanation.'
    AS full_prompt
  FROM medallists_batch
),
raw_response AS (
  -- Step 4: Call Databricks-hosted Llama 3.1 8B
  SELECT ai_query('databricks-meta-llama-3-1-8b-instruct', full_prompt) AS jsonl_text
  FROM prompt
),
split_lines AS (
  -- Step 5: Parse multiline JSONL response
  SELECT trim(line) AS line
  FROM raw_response
  LATERAL VIEW explode(split(jsonl_text, '\n')) AS line
  WHERE trim(line) != ''
),
parsed AS (
  -- Step 6: Extract question/answer fields
  SELECT
    get_json_object(line, '$.input')  AS instruction,
    get_json_object(line, '$.output') AS response
  FROM split_lines
  WHERE line LIKE '{%'
    AND get_json_object(line, '$.input')  IS NOT NULL
    AND get_json_object(line, '$.output') IS NOT NULL
)
-- Step 7: Insert into training table
INSERT INTO olympics_ft_training
SELECT instruction, response, 'medallists' AS source_table
FROM parsed;
```

#### Training Data Generation Queries

Seven SQL queries (in `queries/` directory) generate diverse Q&A pairs:

1. **Create QA Pair for athlets.sql** - 13 athlete fields (height, weight, disciplines, nationality)
2. **create medalist batch.sql** - Medal winners, dates, countries
3. **Create events batch.sql** - Sports, event tags, sport codes
4. **CreACate schedules batch.sql** - Event timing and venues
5. **Create teams batch.sql** - Team compositions, genders, countries
6. **Create_medal_batch.sql** - Country medal totals
7. **athlete_batch_temp table.sql** - Temporary athlete batch processing

**Run iteratively** to generate ~240 training samples total.

#### Output Format

Training table `olympics_ft_training` schema:
```sql
CREATE TABLE olympics_ft_training (
  instruction STRING,   -- User question
  response STRING,      -- Assistant answer
  source_table STRING   -- Origin table for traceability
);
```

Split into train/validation using `split train validation data.sql`.

------------------------------------------------------------------------

## 4. JSONL Training Format for Llama 3 8B

### Format Conversion

The training data must be converted from Databricks table format to Bedrock-compatible JSONL.

**Conversion notebooks**:
- `Convert delta table to jsonl.ipynb` - Basic conversion
- `convert delta table to jsonl chat.ipynb` - Chat format conversion

**Input format** (from Databricks table):
```json
{"input": "Who won the Women's 400m Gold medal?", "output": "PAULINO Marileidy won the Women's 400m Gold medal."}
```

**Output format** (Bedrock chat JSONL):
```json
{"messages":[
  {"role":"user","content":"Who won the Women's 400m Gold medal?"},
  {"role":"assistant","content":"PAULINO Marileidy won the Women's 400m Gold medal."}
]}
```

### Training Files

Generated files:
- **paris2024_ft.jsonl** (23 KB) - Raw Q&A pairs
- **paris2024_ft_chat.jsonl** (37 KB) - Chat-formatted version for Bedrock

**File structure**:
```
s3://bedrockfinetunesk/ft/paris2024/data/
  ├── train.jsonl       # Training set (~240 samples)
  └── val.jsonl         # Validation set (optional)
```

### Sample Q&A Pairs by Category

**Athletes**:
```json
{"messages":[
  {"role":"user","content":"What disciplines does Simone BILES compete in?"},
  {"role":"assistant","content":"Simone BILES competes in Artistic Gymnastics."}
]}
```

**Medallists**:
```json
{"messages":[
  {"role":"user","content":"Which country won the most gold medals?"},
  {"role":"assistant","content":"The United States won 40 gold medals."}
]}
```

**Events**:
```json
{"messages":[
  {"role":"user","content":"What sport code is used for Basketball 3x3?"},
  {"role":"assistant","content":"The sport code for Basketball 3x3 is BK3."}
]}
```

This chat format is required by Bedrock for **instruction fine-tuning** of Llama 3 models.

------------------------------------------------------------------------

## 5. AWS Bedrock Fine-Tuning

### Prerequisites: IAM Configuration

Before fine-tuning, configure IAM role with necessary permissions:

**Trust policy** (`BedrockLlama3FineTuneRole_trust_policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "bedrock.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

**IAM policy** (`BedrockLlama3FineTuneRole_policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::bedrockfinetunesk",
        "arn:aws:s3:::bedrockfinetunesk/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

**Role ARN**: `arn:aws:iam::926643152554:role/BedrockLlama3FineTuneRole`

### Training Configuration

**Hyperparameters** (optimized for small dataset):
- **Base model**: `meta.llama3-8b-instruct-v1:0`
- **Epochs**: 2 (prevents overfitting on ~240 samples)
- **Batch size**: 2 (memory-efficient)
- **Learning rate**: 2e-5 (conservative for instruction tuning)
- **LR scheduler**: linear (gradual decay)
- **Training mechanism**: LoRA (Low-Rank Adaptation) - parameter-efficient fine-tuning

### Fine-Tuning Workflow

#### Step 1: Initiate Training Job

**File**: `FineTune_aws_bedrock_llama3-8b-paris2024.py`

```python
import boto3
import json

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
        "s3Uri": "s3://bedrockfinetunesk/ft/paris2024/output/"
    }
)

print("Started job:", response["jobArn"])
```

**What happens**:
- Bedrock provisions GPU instances automatically
- Downloads training data from S3
- Applies LoRA adapters to Llama 3 8B base model
- Trains for 2 epochs (~5-15 minutes depending on queue)
- Writes fine-tuned model artifacts to output S3 location

#### Step 2: Monitor Training Progress

**File**: `FineTune_poll_job.py`

```python
import boto3
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
```

**Job states**:
- `InProgress` - Training ongoing
- `Completed` - Success, model ready
- `Failed` - Check CloudWatch logs for errors
- `Stopped` - Manually terminated

#### Step 3: Provision Throughput

**File**: `provision_1_TPS_aws_bedrock.py`

Custom models require **provisioned throughput** (cannot use on-demand).

```python
import boto3

bedrock = boto3.client("bedrock", region_name="us-east-1")

response = bedrock.create_provisioned_model_throughput(
    modelUnits=1,  # 1 TPS (tokens per second) - minimum/cheapest
    provisionedModelName="llama3-8b-paris2024-throughput",
    modelId="arn:aws:bedrock:us-east-1:926643152554:custom-model/llama3-8b-paris2024"
)

print(response)
```

**Pricing note**: 1 TPS ≈ $10-15/hour (check AWS pricing). Use `scale_to_zero` in production or delete when not in use.

### Training Time & Costs

- **Training duration**: ~10-20 minutes for 240 samples, 2 epochs
- **Training cost**: ~$5-10 per job (GPU time)
- **Provisioned throughput**: ~$10/hour for 1 TPS (billed hourly)

**Cost optimization**: Delete provisioned throughput when not actively testing.

------------------------------------------------------------------------

## 6. Deployment & Inference

### Inference Script

**File**: `infrence_custome_bedrock_model_paris2024.py`

```python
import boto3
import json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Use chat messages format (not raw prompt)
resp = bedrock_runtime.invoke_model(
    modelId="arn:aws:bedrock:us-east-1:926643152554:custom-model/llama3-8b-paris2024",
    body=json.dumps({
        "messages": [
            {"role": "user", "content": "How many medals did France win at Paris 2024?"}
        ]
    })
)

result = json.loads(resp["body"].read())
print(result)
```

**Key differences from base model inference**:
- **modelId**: Custom model ARN (not base model ID)
- **Input format**: Chat messages (same as training format)
- **No system prompt needed**: Model already tuned for Olympics Q&A

### Testing the Fine-Tuned Model

**Sample queries**:

```python
questions = [
    "Who won the Women's 400m Gold medal?",
    "What disciplines does Simone BILES compete in?",
    "Which country won the most gold medals?",
    "What is the sport code for Basketball 3x3?",
    "When did the Men's Marathon take place?",
    "How many athletes competed in Swimming events?"
]

for question in questions:
    resp = bedrock_runtime.invoke_model(
        modelId=CUSTOM_MODEL_ARN,
        body=json.dumps({
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.2,      # Low temp for factual accuracy
            "max_gen_len": 256,      # Limit response length
            "top_p": 0.9
        })
    )
    answer = json.loads(resp["body"].read())["generation"]
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Performance Expectations

**Accuracy improvements**:
- ✅ **Base Llama 3 8B**: Cannot answer Paris 2024 questions (knowledge cutoff: Jan 2024)
- ✅ **Fine-tuned model**: Accurate answers about medallists, events, schedules, venues
- ✅ **Response time**: ~1-3 seconds at 1 TPS throughput

**Limitations**:
- Only knows data from training set (~240 Q&A pairs)
- May struggle with complex multi-hop reasoning
- Best for factual recall questions

### Integration Patterns

#### REST API Endpoint
```python
# Wrap Bedrock call in FastAPI/Flask for web access
from fastapi import FastAPI

app = FastAPI()

@app.post("/ask")
def ask_olympics(question: str):
    resp = bedrock_runtime.invoke_model(
        modelId=CUSTOM_MODEL_ARN,
        body=json.dumps({"messages": [{"role": "user", "content": question}]})
    )
    return {"answer": json.loads(resp["body"].read())["generation"]}
```

#### Databricks MLflow Integration

**File**: `MLFlow_gen-ai.py` demonstrates MLflow tracking with Databricks models:

```python
import mlflow
from databricks.sdk import WorkspaceClient

mlflow.openai.autolog()  # Auto-log all LLM calls
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

response = client.chat.completions.create(
    model="databricks-llama-4-maverick",  # Or fine-tuned model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Virginia's demography"}
    ]
)
```

**Benefits**:
- Automatic experiment tracking
- Version control for models
- A/B testing between base and fine-tuned models

------------------------------------------------------------------------

## 7. Complete End-to-End Workflow

### Execution Steps (Reproducible Pipeline)

#### Phase 1: Data Engineering (Databricks)

1. **Upload dataset**:
   ```bash
   # Unzip paris-dataset.zip containing 14 CSV files
   unzip paris-dataset.zip
   ```

2. **Load into Delta tables**:
   ```sql
   -- Create database
   CREATE CATALOG IF NOT EXISTS workspace;
   CREATE SCHEMA IF NOT EXISTS workspace.sk_paris_olympics_2024;

   -- Load CSVs to Delta tables (via Databricks UI or COPY INTO)
   COPY INTO workspace.sk_paris_olympics_2024.athletes
   FROM '/FileStore/paris-dataset/athletes.csv'
   FILEFORMAT = CSV
   FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true');

   -- Repeat for: medallists, medals_total, events, schedules, teams, venues
   ```

3. **Create training data table**:
   ```bash
   # Run from queries/ directory
   databricks sql execute -f "Create training data table.sql"
   ```

4. **Generate Q&A pairs** (run iteratively to build dataset):
   ```bash
   # Generate ~30-40 samples per run, repeat 6-8 times
   databricks sql execute -f "Create QA Pair for athlets.sql"
   databricks sql execute -f "create medalist batch.sql"
   databricks sql execute -f "Create events batch.sql"
   databricks sql execute -f "CreACate schedules batch.sql"
   databricks sql execute -f "Create teams batch.sql"
   databricks sql execute -f "Create_medal_batch.sql"
   ```

5. **Split train/validation**:
   ```bash
   databricks sql execute -f "split train validation data.sql"
   ```

6. **Export to JSONL**:
   ```python
   # Run notebooks
   # 1. Convert delta table to jsonl.ipynb
   # 2. convert delta table to jsonl chat.ipynb
   # Output: paris2024_ft_chat.jsonl
   ```

#### Phase 2: AWS Bedrock Fine-Tuning

7. **Upload to S3**:
   ```bash
   aws s3 cp paris2024_ft_chat.jsonl s3://bedrockfinetunesk/ft/paris2024/data/train.jsonl
   aws s3 cp paris2024_ft_chat_val.jsonl s3://bedrockfinetunesk/ft/paris2024/data/val.jsonl
   ```

8. **Configure IAM** (one-time setup):
   ```bash
   aws iam create-role --role-name BedrockLlama3FineTuneRole \
     --assume-role-policy-document file://BedrockLlama3FineTuneRole_trust_policy.json

   aws iam put-role-policy --role-name BedrockLlama3FineTuneRole \
     --policy-name BedrockFineTunePolicy \
     --policy-document file://BedrockLlama3FineTuneRole_policy.json
   ```

9. **Start fine-tuning**:
   ```bash
   python FineTune_aws_bedrock_llama3-8b-paris2024.py
   ```

10. **Monitor job**:
    ```bash
    python FineTune_poll_job.py
    # Wait for "Completed" status (~10-20 minutes)
    ```

11. **Provision throughput**:
    ```bash
    python provision_1_TPS_aws_bedrock.py
    # Wait 5-10 minutes for endpoint provisioning
    ```

#### Phase 3: Testing & Deployment

12. **Test inference**:
    ```bash
    python infrence_custome_bedrock_model_paris2024.py
    ```

13. **Deploy REST API** (optional):
    ```python
    # Wrap in FastAPI/Flask for production access
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```

### Repository Structure

```
Data-bricks-AI-Experiment/
├── paris-dataset/              # Source CSV files (14 files)
│   ├── athletes.csv
│   ├── medallists.csv
│   ├── events.csv
│   └── ...
├── queries/                    # SQL Q&A generation queries
│   ├── Create QA Pair for athlets.sql
│   ├── create medalist batch.sql
│   ├── Create events batch.sql
│   └── ...
├── FineTune_aws_bedrock_llama3-8b-paris2024.py   # Start training job
├── FineTune_poll_job.py                           # Monitor training
├── provision_1_TPS_aws_bedrock.py                 # Provision throughput
├── infrence_custome_bedrock_model_paris2024.py    # Test inference
├── MLFlow_gen-ai.py                               # MLflow integration demo
├── Convert delta table to jsonl.ipynb             # Export to JSONL
├── convert delta table to jsonl chat.ipynb        # Chat format conversion
├── Build your first AI agent.ipynb                # Databricks Agent Framework
├── paris2024_ft.jsonl                             # Generated Q&A pairs
├── paris2024_ft_chat.jsonl                        # Chat-formatted JSONL
├── BedrockLlama3FineTuneRole_policy.json         # IAM permissions
└── BedrockLlama3FineTuneRole_trust_policy.json   # IAM trust relationship
```

------------------------------------------------------------------------

## 8. Skills Demonstrated

### Data Engineering
- ✅ **ETL pipelines**: CSV ingestion → Delta tables → JSONL export
- ✅ **Databricks SQL**: Complex CTEs, JSON manipulation, `ai_query()` integration
- ✅ **Data quality**: Filtering nulls, randomization, train/val splitting
- ✅ **Unity Catalog**: Multi-catalog data organization

### Machine Learning Operations (MLOps)
- ✅ **LLM fine-tuning**: Hyperparameter selection, LoRA adaptation
- ✅ **Synthetic data generation**: LLM-to-LLM transformation pipeline
- ✅ **Model versioning**: Custom model naming, ARN management
- ✅ **MLflow tracking**: Experiment logging, model registry (Databricks)

### Cloud Engineering (AWS)
- ✅ **IAM configuration**: Least-privilege policies, service roles
- ✅ **S3 data lakes**: Structured data storage, versioning
- ✅ **Bedrock API**: boto3 SDK, model customization jobs
- ✅ **Cost optimization**: Minimal TPS provisioning, resource cleanup

### Software Engineering
- ✅ **Python**: boto3, databricks-sdk, MLflow, API development
- ✅ **SQL**: Advanced analytics, JSON parsing, CTE patterns
- ✅ **API design**: REST endpoints, chat message formatting
- ✅ **Documentation**: Architecture diagrams, reproducible workflows

### GenAI Architecture
- ✅ **Instruction tuning**: Chat format, role-based conversations
- ✅ **Prompt engineering**: Constrained generation, output format control
- ✅ **Model deployment**: Provisioned throughput, endpoint management
- ✅ **RAG alternatives**: When fine-tuning beats retrieval-augmented generation

------------------------------------------------------------------------

## 9. Alternative: Databricks Agent Framework

This repository also includes **Build your first AI agent.ipynb**, demonstrating Mosaic AI Agent Framework:

### Key Concepts

- **ResponsesAgent interface**: MLflow standard for chat agents
- **Tool integration**: Built-in `system.ai.python_exec` for code execution
- **Unity Catalog tools**: Custom UC functions as agent tools
- **Deployment**: One-click endpoint deployment via `agents.deploy()`

### When to Use Agent Framework vs Fine-Tuning

| Approach | Best For | Trade-offs |
|----------|----------|------------|
| **Fine-Tuning** | Static knowledge, factual recall, consistent responses | Requires training data, less flexible |
| **Agent Framework** | Dynamic queries, tool use, multi-step reasoning | Higher latency, more complex architecture |

**Example use case**: Combine both approaches
- Fine-tune model on Olympics facts
- Deploy as agent with tools for real-time stats lookup

------------------------------------------------------------------------

## 10. Production Considerations

### Monitoring & Observability
- **CloudWatch logs**: Track Bedrock job failures, latency spikes
- **MLflow traces**: Debug agent tool calls, LLM responses
- **Custom metrics**: Question type distribution, answer accuracy

### Security & Compliance
- **IAM least privilege**: Separate roles for training vs inference
- **VPC endpoints**: Private Bedrock access (no internet routing)
- **Data privacy**: Training data remains in your AWS account

### Scaling & Performance
- **Increase TPS**: Scale from 1 → 10 TPS for higher throughput
- **Batch inference**: Process multiple queries in single API call
- **Caching**: Redis/ElastiCache for frequently asked questions

### Cost Management
```python
# Automated cost control
def cleanup_unused_resources():
    # Delete provisioned throughput when idle
    if is_idle_for_hours(2):
        bedrock.delete_provisioned_model_throughput(
            provisionedModelId=THROUGHPUT_ARN
        )
```

------------------------------------------------------------------------

## 11. Final Result

A **production-ready fine-tuned Llama 3 8B model** that:
- ✅ Accurately answers Paris 2024 Olympics questions (beyond base model's knowledge cutoff)
- ✅ Deploys via managed AWS Bedrock infrastructure (no GPU management)
- ✅ Costs ~$10/hour at minimal throughput (deleteable when not in use)
- ✅ Integrates with Databricks MLflow for experiment tracking

### Interview Showcase Value

This project demonstrates:
1. **End-to-end ML pipeline**: Data → Training → Deployment
2. **Cloud-native architecture**: Serverless compute, managed services
3. **Cost consciousness**: Minimal resources, cleanup automation
4. **Production readiness**: IAM security, monitoring, reproducibility
5. **Innovation**: Zero manual labeling via LLM-powered transformation

**Perfect for roles in**:
- Machine Learning Engineer
- Data Engineer
- MLOps / LLMOps Engineer
- GenAI Solutions Architect
- Applied AI Researcher
