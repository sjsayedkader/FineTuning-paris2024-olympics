import mlflow
from databricks.sdk import WorkspaceClient

# Set these environment variables before running:
# $Env:DATABRICKS_TOKEN="your-databricks-token"
# $Env:DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
# $Env:MLFLOW_TRACKING_URI="databricks"
# $Env:MLFLOW_REGISTRY_URI="databricks-uc"
# $Env:MLFLOW_EXPERIMENT_ID="your-experiment-id"

mlflow.openai.autolog()
#mlflow.anthropic.autolog()
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

response = client.chat.completions.create(
  # You can replace 'model' with any Databricks hosted model from here: http://<workspace-url>/ml/endpoints
  model="databricks-llama-4-maverick",
  messages=[
    {
      "role": "system", 
      "content": "You are a helpful assistant.",
    },
    {
      "role": "user",
      "content": "explain virginia's demography",
    },
  ],
)




