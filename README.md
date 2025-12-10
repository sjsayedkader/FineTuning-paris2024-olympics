# Fine-Tuning Llama 3 8B for Paris 2024 Olympics Q&A

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![Databricks](https://img.shields.io/badge/Databricks-Delta-red.svg)](https://www.databricks.com/)

An end-to-end LLM fine-tuning pipeline that teaches Meta Llama 3 8B Instruct updated knowledge about the Paris 2024 Olympics using **Databricks**, **AWS Bedrock**, and **synthetic Q&A generation**.

## 🎯 Project Overview

This project demonstrates how to:
- ✅ Ingest structured Olympic data into Databricks Delta tables
- ✅ Auto-generate 240+ training Q&A pairs using LLM-to-LLM transformation
- ✅ Fine-tune Llama 3 8B via AWS Bedrock model customization
- ✅ Deploy a custom model endpoint for inference

**Key Innovation**: Zero manual data labeling - training data generated entirely through Databricks `ai_query()` with Llama 3.1 8B.

## 📖 Documentation

For comprehensive details, see **[Paris2024_Llama3_FT_Detailed_Showcase.md](Paris2024_Llama3_FT_Detailed_Showcase.md)** which covers:
- Architecture diagram
- Step-by-step data engineering workflow
- SQL patterns for Q&A generation
- AWS Bedrock fine-tuning configuration
- Deployment and inference examples
- Production considerations

For quick reference, see **[CLAUDE.md](CLAUDE.md)** for repository structure and common commands.

## 🏗️ Architecture

```
Kaggle CSVs → Databricks Delta → ai_query() Q&A Generation →
JSONL Dataset → S3 → Bedrock Fine-Tuning → Custom Model Endpoint
```

## 📊 Dataset

Paris 2024 Olympics data (14 CSV files):
- **athletes.csv** - Athlete profiles, disciplines, nationalities
- **medallists.csv** - Medal winners by event
- **events.csv** - Competition events and sports codes
- **schedules.csv** - Event timing and venues
- **teams.csv** - Team compositions
- **medals_total.csv** - Country medal counts

## 🚀 Quick Start

### Prerequisites
- Databricks workspace with Unity Catalog
- AWS account with Bedrock access
- Python 3.8+
- boto3, databricks-sdk, mlflow

### 1. Data Engineering (Databricks)

```bash
# Load CSVs into Delta tables
# Run SQL queries in queries/ directory to generate Q&A pairs
databricks sql execute -f "queries/Create QA Pair for athlets.sql"
databricks sql execute -f "queries/create medalist batch.sql"
# ... repeat for other queries

# Export to JSONL
jupyter notebook "Convert delta table to jsonl chat.ipynb"
```

### 2. Fine-Tuning (AWS Bedrock)

```bash
# Upload training data to S3
aws s3 cp paris2024_ft_chat.jsonl s3://your-bucket/data/train.jsonl

# Configure IAM role (one-time)
aws iam create-role --role-name BedrockLlama3FineTuneRole \
  --assume-role-policy-document file://BedrockLlama3FineTuneRole_trust_policy.json

# Start fine-tuning
python FineTune_aws_bedrock_llama3-8b-paris2024.py

# Monitor progress
python FineTune_poll_job.py

# Provision throughput
python provision_1_TPS_aws_bedrock.py
```

### 3. Inference

```python
import boto3
import json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

resp = bedrock_runtime.invoke_model(
    modelId="arn:aws:bedrock:us-east-1:ACCOUNT:custom-model/llama3-8b-paris2024",
    body=json.dumps({
        "messages": [
            {"role": "user", "content": "Who won the Women's 400m Gold medal?"}
        ]
    })
)

print(json.loads(resp["body"].read())["generation"])
```

## 📁 Repository Structure

```
├── paris-dataset/                # Source CSV files
├── queries/                      # SQL Q&A generation queries
├── FineTune_aws_bedrock_llama3-8b-paris2024.py
├── FineTune_poll_job.py
├── provision_1_TPS_aws_bedrock.py
├── infrence_custome_bedrock_model_paris2024.py
├── MLFlow_gen-ai.py              # MLflow integration example
├── Build your first AI agent.ipynb   # Databricks Agent Framework
├── paris2024_ft.jsonl            # Generated Q&A pairs
├── paris2024_ft_chat.jsonl       # Chat-formatted training data
├── BedrockLlama3FineTuneRole_policy.json
└── Paris2024_Llama3_FT_Detailed_Showcase.md
```

## 🛠️ Technologies

- **Databricks**: Delta tables, `ai_query()`, Unity Catalog
- **AWS Bedrock**: Model customization, provisioned throughput
- **MLflow**: Experiment tracking, model registry
- **Python**: boto3, databricks-sdk, pandas

## 💡 Key Learnings

### LLM-to-LLM Transformation Pattern
Using one LLM to transform structured data into training examples for another LLM eliminates manual labeling:

```sql
SELECT ai_query('databricks-meta-llama-3-1-8b-instruct',
  'Generate Q&A pairs from: ' || to_json(collect_list(struct(*))))
FROM sample_data;
```

### LoRA Fine-Tuning
Bedrock uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:
- 2 epochs, batch size 2, learning rate 2e-5
- ~10-20 minutes training time
- ~$5-10 per training job

### Cost Optimization
- Minimal provisioned throughput: 1 TPS (~$10/hour)
- Delete throughput when not in use
- Serverless Databricks inference for Q&A generation

## 📈 Results

**Before fine-tuning**: Base Llama 3 8B cannot answer Paris 2024 questions (knowledge cutoff: Jan 2024)

**After fine-tuning**: Accurate responses about:
- Medal winners by event
- Athlete nationalities and disciplines
- Country medal counts
- Event schedules and venues

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Expand training dataset (more SQL queries)
- Add validation metrics
- Implement A/B testing framework
- Create REST API wrapper

## 📄 License

This project is for educational and demonstration purposes.

## 🔗 References

- [AWS Bedrock Model Customization](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization.html)
- [Databricks ai_query()](https://docs.databricks.com/en/large-language-models/ai-functions.html)
- [Meta Llama 3](https://llama.meta.com/)
- [Kaggle Paris 2024 Olympics Dataset](https://www.kaggle.com/datasets/piterfm/paris-2024-olympic-summer-games)

## 👤 Author

**Sayed Kader**
- GitHub: [@sjsayedkader](https://github.com/sjsayedkader)
- Project: [FineTuning-paris2024-olympics](https://github.com/sjsayedkader/FineTuning-paris2024-olympics)

---

⭐ If you find this project useful, please star the repository!
