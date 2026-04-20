package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1263,
			Title:       "Azure AI and Machine Learning Services",
			Description: "Explore Azure AI services including Cognitive Services, Azure Machine Learning, OpenAI Service, and AI-powered application development.",
			Order:       63,
			Lessons: []problems.Lesson{
				{
					Title: "Azure AI and Cognitive Services",
					Content: `Azure provides pre-built AI services and a full ML platform for custom model development.

**Azure AI Services (Cognitive Services):**
` + "```" + `
Vision services:
  Computer Vision:
    - Image analysis (tags, categories, objects)
    - OCR (printed and handwritten text)
    - Spatial analysis (people counting)
    - Image thumbnails
  
  Custom Vision:
    - Custom image classification
    - Custom object detection
    - Few-shot learning
  
  Face API:
    - Face detection and recognition
    - Face verification
    - Emotion/attribute detection
  
  Document Intelligence (Form Recognizer):
    - Document extraction
    - Receipt, invoice, ID processing
    - Custom document models

Language services:
  Language Understanding:
    - Text analytics (sentiment, key phrases)
    - Named entity recognition
    - Language detection
    - Summarization
    - Question answering
  
  Translator:
    - 100+ languages
    - Real-time translation
    - Custom terminology
    - Document translation
  
  Immersive Reader:
    - Read-aloud
    - Visual dictionary
    - Syllable highlighting

Speech services:
  Speech-to-Text:
    - Real-time transcription
    - Batch transcription
    - Custom speech models
    - Multiple languages
  
  Text-to-Speech:
    - Neural voices
    - Custom Neural Voice
    - SSML support
    - Audio content creation
  
  Speech Translation:
    - Real-time speech translation
    - Multiple language pairs

Create AI service:
  # Multi-service account
  az cognitiveservices account create \
    --name myai -g myRG --kind CognitiveServices \
    --sku S0 --location eastus --yes
  
  # Single service
  az cognitiveservices account create \
    --name myvision -g myRG --kind ComputerVision \
    --sku S1 --location eastus --yes
  
  # Get keys
  az cognitiveservices account keys list \
    --name myai -g myRG
  
  # Get endpoint
  az cognitiveservices account show \
    --name myai -g myRG --query "properties.endpoint" -o tsv

Usage examples (curl):
  # Sentiment analysis
  curl -X POST "$ENDPOINT/text/analytics/v3.1/sentiment" \
    -H "Ocp-Apim-Subscription-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "documents": [
        {"id": "1", "language": "en", "text": "Azure is amazing!"},
        {"id": "2", "language": "en", "text": "The service had issues."}
      ]
    }'
  
  # Image analysis
  curl -X POST "$ENDPOINT/vision/v3.2/analyze?visualFeatures=Tags,Objects,Description" \
    -H "Ocp-Apim-Subscription-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com/image.jpg"}'
  
  # Speech to text
  curl -X POST "$ENDPOINT/speechtotext/v3.1/transcriptions" \
    -H "Ocp-Apim-Subscription-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "contentUrls": ["https://mystorageacct.blob.core.windows.net/audio/meeting.wav"],
      "locale": "en-US",
      "displayName": "Meeting transcription"
    }'
` + "```" + `

**Azure OpenAI Service:**
` + "```" + `
Access to OpenAI models on Azure infrastructure.

Available models:
  GPT-4o:           Latest multimodal model
  GPT-4 Turbo:      128K context, vision
  GPT-4:            8K/32K context
  GPT-3.5 Turbo:    Fast, cost-effective
  DALL-E 3:         Image generation
  Whisper:          Speech recognition
  Text Embedding:   ada-002, text-embedding-3-*

Create Azure OpenAI:
  az cognitiveservices account create \
    --name myopenai -g myRG \
    --kind OpenAI --sku S0 \
    --location eastus \
    --custom-domain myopenai

  # Deploy model
  az cognitiveservices account deployment create \
    --name myopenai -g myRG \
    --deployment-name gpt4 \
    --model-name gpt-4o \
    --model-version "2024-05-13" \
    --model-format OpenAI \
    --sku-capacity 10 --sku-name Standard

Chat completion:
  curl "https://myopenai.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-02-01" \
    -H "Content-Type: application/json" \
    -H "api-key: $AZURE_OPENAI_KEY" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Azure?"}
      ],
      "temperature": 0.7,
      "max_tokens": 500
    }'

Embeddings:
  curl "https://myopenai.openai.azure.com/openai/deployments/embedding/embeddings?api-version=2024-02-01" \
    -H "Content-Type: application/json" \
    -H "api-key: $AZURE_OPENAI_KEY" \
    -d '{
      "input": ["Azure cloud computing platform"]
    }'

Content filtering:
  - Built-in content safety filters
  - Categories: hate, sexual, violence, self-harm
  - Severity levels: low, medium, high
  - Custom filters possible
  - Prompt shields (jailbreak detection)

RAG (Retrieval-Augmented Generation):
  Azure AI Search + Azure OpenAI:
  
  1. Index documents in Azure AI Search
  2. User query → Search for relevant docs
  3. Include docs as context in prompt
  4. GPT generates answer grounded in data
  
  "On Your Data" feature:
    curl "$ENDPOINT/openai/deployments/gpt4/chat/completions?api-version=2024-02-01" \
      -H "api-key: $KEY" -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "What is our refund policy?"}],
        "data_sources": [{
          "type": "azure_search",
          "parameters": {
            "endpoint": "https://mysearch.search.windows.net",
            "index_name": "company-docs",
            "authentication": {"type": "api_key", "key": "$SEARCH_KEY"}
          }
        }]
      }'
` + "```" + ``,
					CodeExamples: `# Azure AI service scripts

# 1. AI service inventory
#!/bin/bash
echo "=== Azure AI Services Inventory ==="

for acct in $(az cognitiveservices account list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az cognitiveservices account list \
        --query "[?name=='$acct'].resourceGroup" -o tsv | head -1)
    
    KIND=$(az cognitiveservices account show -n "$acct" -g "$RG" \
        --query "kind" -o tsv 2>/dev/null)
    SKU=$(az cognitiveservices account show -n "$acct" -g "$RG" \
        --query "sku.name" -o tsv 2>/dev/null)
    ENDPOINT=$(az cognitiveservices account show -n "$acct" -g "$RG" \
        --query "properties.endpoint" -o tsv 2>/dev/null)
    
    echo "  $acct ($KIND, $SKU)"
    echo "    Endpoint: $ENDPOINT"
    
    # Deployments (for OpenAI)
    if [ "$KIND" = "OpenAI" ]; then
        echo "    Deployments:"
        az cognitiveservices account deployment list \
            -n "$acct" -g "$RG" \
            --query "[].{name:name, model:properties.model.name, version:properties.model.version}" \
            -o table 2>/dev/null
    fi
done

# 2. OpenAI usage reporter
#!/bin/bash
echo "=== Azure OpenAI Usage ==="

ACCOUNT="${1:-myopenai}"
RG="${2:-myRG}"

echo "Account: $ACCOUNT"

# List deployments
echo ""
echo "--- Model Deployments ---"
az cognitiveservices account deployment list \
    -n "$ACCOUNT" -g "$RG" \
    --query "[].{
        name:name,
        model:properties.model.name,
        version:properties.model.version,
        capacity:sku.capacity,
        status:properties.provisioningState
    }" -o table 2>/dev/null

# Usage metrics
echo ""
echo "--- Usage Metrics (Last 24h) ---"
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_TIME=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-24H +%Y-%m-%dT%H:%M:%SZ)

RESOURCE_ID=$(az cognitiveservices account show -n "$ACCOUNT" -g "$RG" --query "id" -o tsv 2>/dev/null)

# Token usage
az monitor metrics list \
    --resource "$RESOURCE_ID" \
    --metric "TokenTransaction" \
    --start-time "$START_TIME" --end-time "$END_TIME" \
    --interval PT1H \
    --query "value[0].timeseries[0].data[].{time:timeStamp, total:total}" \
    -o table 2>/dev/null

# 3. Content safety checker
#!/bin/bash
echo "=== Content Safety Check ==="

ENDPOINT="${1}"
KEY="${2}"
TEXT="${3:-This is a test message}"

if [ -z "$ENDPOINT" ] || [ -z "$KEY" ]; then
    echo "Usage: $0 <endpoint> <api-key> [text]"
    exit 1
fi

# Text analysis
curl -s -X POST "$ENDPOINT/contentsafety/text:analyze?api-version=2023-10-01" \
    -H "Ocp-Apim-Subscription-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"text\": \"$TEXT\",
        \"categories\": [\"Hate\", \"Violence\", \"Sexual\", \"SelfHarm\"],
        \"outputType\": \"FourSeverityLevels\"
    }" | jq '{
        hate: .categoriesAnalysis[] | select(.category == "Hate") | .severity,
        violence: .categoriesAnalysis[] | select(.category == "Violence") | .severity,
        sexual: .categoriesAnalysis[] | select(.category == "Sexual") | .severity,
        selfHarm: .categoriesAnalysis[] | select(.category == "SelfHarm") | .severity
    }' 2>/dev/null`,
				},
				{
					Title: "Azure Machine Learning Platform",
					Content: `Azure Machine Learning provides an end-to-end platform for building, training, and deploying ML models.

**Azure ML Workspace:**
` + "```" + `
Components:
  Workspace:      Central resource for ML assets
  Compute:        Training and inference compute
  Datastores:     Data source connections
  Datasets:       Versioned data references
  Experiments:    Training run tracking
  Models:         Registered model versions
  Endpoints:      Deployment targets
  Pipelines:      ML workflow automation
  Environments:   Docker + conda specs

Create workspace:
  az extension add -n ml
  
  az ml workspace create \
    --name mymlws -g myRG \
    --location eastus
  
  # With custom settings
  az ml workspace create \
    --name mymlws -g myRG \
    --storage-account mystorageacct \
    --key-vault mykeyvault \
    --container-registry myacr \
    --application-insights myappinsights

Compute:
  # Compute instance (dev/notebook)
  az ml compute create \
    --name dev-vm --type ComputeInstance \
    --size Standard_DS3_v2 \
    --workspace-name mymlws -g myRG
  
  # Compute cluster (training)
  az ml compute create \
    --name gpu-cluster --type AmlCompute \
    --size Standard_NC6s_v3 \
    --min-instances 0 --max-instances 4 \
    --idle-time-before-scale-down 120 \
    --workspace-name mymlws -g myRG
  
  # Kubernetes cluster (inference)
  az ml compute attach \
    --name aks-inference --type Kubernetes \
    --resource-id "/subscriptions/.../managedClusters/myaks" \
    --workspace-name mymlws -g myRG

Data:
  # Register datastore
  az ml datastore create \
    --name mydatastore --type azure_blob \
    --account-name mystorageacct \
    --container-name ml-data \
    --workspace-name mymlws -g myRG
  
  # Create data asset
  az ml data create \
    --name training-data --version 1 \
    --path azureml://datastores/mydatastore/paths/training/ \
    --type uri_folder \
    --workspace-name mymlws -g myRG
` + "```" + `

**Training and MLOps:**
` + "```" + `
Training job (command):
  # job.yml
  $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
  type: command
  
  experiment_name: my-experiment
  compute: azureml:gpu-cluster
  
  code: ./src
  command: >
    python train.py
      --data ${{inputs.training_data}}
      --learning-rate ${{inputs.lr}}
      --epochs ${{inputs.epochs}}
  
  inputs:
    training_data:
      type: uri_folder
      path: azureml:training-data@latest
    lr: 0.001
    epochs: 50
  
  environment:
    image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04
    conda_file: conda.yml
  
  # Submit
  az ml job create --file job.yml --workspace-name mymlws -g myRG

Sweep job (hyperparameter tuning):
  $schema: https://azuremlschemas.azureedge.net/latest/sweepJob.schema.json
  type: sweep
  
  experiment_name: hyperparam-sweep
  compute: azureml:gpu-cluster
  
  sampling_algorithm: bayesian
  
  search_space:
    learning_rate:
      type: loguniform
      min_value: -7
      max_value: -3
    batch_size:
      type: choice
      values: [16, 32, 64, 128]
  
  objective:
    goal: minimize
    primary_metric: loss
  
  limits:
    max_total_trials: 20
    max_concurrent_trials: 4
    timeout: 7200
  
  trial:
    command: >
      python train.py
        --lr ${{search_space.learning_rate}}
        --batch-size ${{search_space.batch_size}}
    code: ./src
    environment: azureml:my-env@latest

Pipeline:
  $schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
  type: pipeline
  
  experiment_name: my-pipeline
  
  settings:
    default_compute: azureml:gpu-cluster
  
  jobs:
    preprocess:
      type: command
      component: azureml:preprocess@latest
      inputs:
        raw_data:
          type: uri_folder
          path: azureml:raw-data@latest
      outputs:
        processed_data:
          type: uri_folder
    
    train:
      type: command
      component: azureml:train@latest
      inputs:
        training_data: ${{parent.jobs.preprocess.outputs.processed_data}}
      outputs:
        model:
          type: mlflow_model
    
    evaluate:
      type: command
      component: azureml:evaluate@latest
      inputs:
        model: ${{parent.jobs.train.outputs.model}}
        test_data:
          type: uri_folder
          path: azureml:test-data@latest
    
    register:
      type: command
      component: azureml:register@latest
      inputs:
        model: ${{parent.jobs.train.outputs.model}}
        metrics: ${{parent.jobs.evaluate.outputs.metrics}}

Model deployment:
  # Register model
  az ml model create \
    --name my-model --version 1 \
    --path runs:/my-run-id/model \
    --type mlflow_model \
    --workspace-name mymlws -g myRG
  
  # Online endpoint
  az ml online-endpoint create \
    --name my-endpoint \
    --workspace-name mymlws -g myRG
  
  # Deployment
  az ml online-deployment create \
    --name blue --endpoint-name my-endpoint \
    --model azureml:my-model@latest \
    --instance-type Standard_DS3_v2 \
    --instance-count 1 \
    --workspace-name mymlws -g myRG
  
  # Set traffic
  az ml online-endpoint update \
    --name my-endpoint \
    --traffic "blue=100" \
    --workspace-name mymlws -g myRG
  
  # Test
  az ml online-endpoint invoke \
    --name my-endpoint \
    --request-file request.json \
    --workspace-name mymlws -g myRG

Responsible AI:
  - Model interpretability (SHAP, LIME)
  - Fairness assessment
  - Error analysis
  - Counterfactual analysis
  - Data explorer
  - Responsible AI dashboard
` + "```" + ``,
					CodeExamples: `# Azure ML management scripts

# 1. ML workspace overview
#!/bin/bash
echo "=== Azure ML Workspace Overview ==="

for ws in $(az ml workspace list --query "[].name" -o tsv 2>/dev/null); do
    RG=$(az ml workspace list --query "[?name=='$ws'].resourceGroup" -o tsv | head -1)
    echo "Workspace: $ws ($RG)"
    
    # Compute
    echo "  Compute:"
    az ml compute list --workspace-name "$ws" -g "$RG" \
        --query "[].{name:name, type:type, state:state, size:size}" \
        -o table 2>/dev/null
    
    # Models
    echo "  Models:"
    az ml model list --workspace-name "$ws" -g "$RG" \
        --query "[].{name:name, version:version}" \
        -o table 2>/dev/null | head -10
    
    # Endpoints
    echo "  Online Endpoints:"
    az ml online-endpoint list --workspace-name "$ws" -g "$RG" \
        --query "[].{name:name, state:provisioning_state, traffic:traffic}" \
        -o table 2>/dev/null
    
    echo ""
done

# 2. Training job monitor
#!/bin/bash
echo "=== ML Training Jobs ==="

WS="${1:-mymlws}"
RG="${2:-myRG}"

# Recent jobs
echo "--- Recent Jobs ---"
az ml job list --workspace-name "$WS" -g "$RG" \
    --query "[:10].{
        name:name, status:status,
        type:type, experiment:experiment_name,
        compute:compute
    }" -o table 2>/dev/null

# Running jobs
echo ""
echo "--- Running Jobs ---"
az ml job list --workspace-name "$WS" -g "$RG" \
    --query "[?status=='Running'].{
        name:name, experiment:experiment_name,
        compute:compute
    }" -o table 2>/dev/null

# Failed jobs
echo ""
echo "--- Failed Jobs (last 10) ---"
az ml job list --workspace-name "$WS" -g "$RG" \
    --query "[?status=='Failed'][:10].{
        name:name, experiment:experiment_name
    }" -o table 2>/dev/null

# 3. Endpoint health checker
#!/bin/bash
echo "=== ML Endpoint Health ==="

WS="${1:-mymlws}"
RG="${2:-myRG}"

for endpoint in $(az ml online-endpoint list \
    --workspace-name "$WS" -g "$RG" \
    --query "[].name" -o tsv 2>/dev/null); do
    
    echo "Endpoint: $endpoint"
    
    STATE=$(az ml online-endpoint show \
        --name "$endpoint" --workspace-name "$WS" -g "$RG" \
        --query "provisioning_state" -o tsv 2>/dev/null)
    
    SCORING_URI=$(az ml online-endpoint show \
        --name "$endpoint" --workspace-name "$WS" -g "$RG" \
        --query "scoring_uri" -o tsv 2>/dev/null)
    
    echo "  State: $STATE"
    echo "  URI: $SCORING_URI"
    
    # Deployments
    echo "  Deployments:"
    az ml online-deployment list \
        --endpoint-name "$endpoint" --workspace-name "$WS" -g "$RG" \
        --query "[].{
            name:name, model:model,
            instanceType:instance_type, instances:instance_count
        }" -o table 2>/dev/null
    
    # Traffic split
    TRAFFIC=$(az ml online-endpoint show \
        --name "$endpoint" --workspace-name "$WS" -g "$RG" \
        --query "traffic" -o json 2>/dev/null)
    echo "  Traffic: $TRAFFIC"
    echo ""
done`,
				},
			},
		},
	})
}
