package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1253,
			Title:       "Azure DevOps and CI/CD",
			Description: "Master Azure DevOps services, GitHub Actions for Azure, Infrastructure as Code with Terraform and Bicep, and deployment best practices.",
			Order:       53,
			Lessons: []problems.Lesson{
				{
					Title: "Azure DevOps and GitHub Actions",
					Content: `Azure DevOps and GitHub Actions provide comprehensive CI/CD pipelines for building, testing, and deploying to Azure.

**Azure DevOps Services:**
` + "```" + `
Azure DevOps components:
  Azure Repos:      Git repositories
  Azure Pipelines:  CI/CD pipelines
  Azure Boards:     Work tracking (Agile, Scrum, Kanban)
  Azure Test Plans:  Testing tools
  Azure Artifacts:   Package feeds (npm, NuGet, Python, Maven)

Azure Pipelines (YAML):
  # azure-pipelines.yml
  trigger:
    branches:
      include:
      - main
      - release/*
    paths:
      exclude:
      - docs/*
      - README.md
  
  pool:
    vmImage: 'ubuntu-latest'
  
  variables:
    - group: production-vars
    - name: buildConfiguration
      value: 'Release'
  
  stages:
  - stage: Build
    jobs:
    - job: BuildAndTest
      steps:
      - task: UseDotNet@2
        inputs:
          version: '8.0.x'
      
      - script: dotnet build --configuration $(buildConfiguration)
        displayName: 'Build'
      
      - script: dotnet test --configuration $(buildConfiguration) --logger trx
        displayName: 'Test'
      
      - task: PublishTestResults@2
        inputs:
          testResultsFormat: 'VSTest'
          testResultsFiles: '**/*.trx'
      
      - task: Docker@2
        inputs:
          containerRegistry: 'myACR'
          repository: 'myapp'
          command: 'buildAndPush'
          Dockerfile: '**/Dockerfile'
          tags: |
            $(Build.BuildId)
            latest
  
  - stage: Deploy_Staging
    dependsOn: Build
    condition: succeeded()
    jobs:
    - deployment: DeployStaging
      environment: 'staging'
      strategy:
        runOnce:
          deploy:
            steps:
            - task: AzureWebAppContainer@1
              inputs:
                azureSubscription: 'my-azure-connection'
                appName: 'myapp-staging'
                imageName: 'myacr.azurecr.io/myapp:$(Build.BuildId)'
  
  - stage: Deploy_Production
    dependsOn: Deploy_Staging
    condition: succeeded()
    jobs:
    - deployment: DeployProduction
      environment: 'production'
      strategy:
        runOnce:
          deploy:
            steps:
            - task: AzureWebAppContainer@1
              inputs:
                azureSubscription: 'my-azure-connection'
                appName: 'myapp-prod'
                imageName: 'myacr.azurecr.io/myapp:$(Build.BuildId)'
` + "```" + `

**GitHub Actions for Azure:**
` + "```" + `
# .github/workflows/deploy-azure.yml
name: Deploy to Azure

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  id-token: write
  contents: read

env:
  AZURE_WEBAPP_NAME: myapp
  REGISTRY: myacr.azurecr.io
  IMAGE_NAME: myapp

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v5
      with:
        go-version: '1.22'
    
    - name: Test
      run: go test ./...
    
    - name: Azure Login (OIDC)
      uses: azure/login@v2
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    
    - name: ACR Login
      run: az acr login --name myacr
    
    - name: Build and push
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
  
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - name: Azure Login
      uses: azure/login@v2
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    
    - name: Deploy to staging
      uses: azure/webapps-deploy@v3
      with:
        app-name: ${{ env.AZURE_WEBAPP_NAME }}-staging
        images: '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}'
    
    - name: Health check
      run: |
        for i in $(seq 1 30); do
          if curl -sf https://${{ env.AZURE_WEBAPP_NAME }}-staging.azurewebsites.net/health; then
            echo "Staging healthy"
            exit 0
          fi
          sleep 10
        done
        echo "Staging health check failed"
        exit 1
  
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Azure Login
      uses: azure/login@v2
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    
    - name: Deploy to production
      uses: azure/webapps-deploy@v3
      with:
        app-name: ${{ env.AZURE_WEBAPP_NAME }}
        images: '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}'

# AKS deployment workflow
# .github/workflows/deploy-aks.yml
name: Deploy to AKS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Azure Login
      uses: azure/login@v2
      with:
        client-id: ${{ secrets.AZURE_CLIENT_ID }}
        tenant-id: ${{ secrets.AZURE_TENANT_ID }}
        subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    
    - name: Set AKS context
      uses: azure/aks-set-context@v3
      with:
        resource-group: myRG
        cluster-name: myAKS
    
    - name: Deploy to AKS
      uses: azure/k8s-deploy@v4
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          myacr.azurecr.io/myapp:${{ github.sha }}
` + "```" + ``,
					CodeExamples: `# Azure DevOps and CI/CD scripts

# 1. Azure Container Registry management
#!/bin/bash
ACR_NAME="${1:?Usage: $0 <acr-name>}"

echo "=== ACR Status: $ACR_NAME ==="

# Registry info
az acr show --name "$ACR_NAME" --query "{
    loginServer:loginServer, sku:sku.name, 
    adminEnabled:adminUserEnabled, publicAccess:publicNetworkAccess
}" -o json | jq .

echo ""
echo "--- Repositories ---"
for repo in $(az acr repository list --name "$ACR_NAME" -o tsv 2>/dev/null); do
    TAG_COUNT=$(az acr repository show-tags --name "$ACR_NAME" \
        --repository "$repo" --query "length(@)" -o tsv 2>/dev/null)
    
    LATEST=$(az acr repository show-tags --name "$ACR_NAME" \
        --repository "$repo" --orderby time_desc --top 1 -o tsv 2>/dev/null)
    
    printf "  %-30s  Tags: %4s  Latest: %s\n" "$repo" "$TAG_COUNT" "$LATEST"
done

# Check for old images
echo ""
echo "--- Cleanup Candidates (>90 days) ---"
for repo in $(az acr repository list --name "$ACR_NAME" -o tsv 2>/dev/null); do
    OLD_TAGS=$(az acr repository show-manifests --name "$ACR_NAME" \
        --repository "$repo" --orderby time_asc \
        --query "[?timestamp < '$(date -d '90 days ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-90d +%Y-%m-%dT%H:%M:%S)'].digest" \
        -o tsv 2>/dev/null | wc -l)
    
    if [ "$OLD_TAGS" -gt 0 ]; then
        echo "  $repo: $OLD_TAGS manifests older than 90 days"
    fi
done

# 2. Infrastructure deployment wrapper
#!/bin/bash
set -euo pipefail

ENV="${1:?Usage: $0 <environment> <action>}"
ACTION="${2:?Usage: $0 <environment> <action>}"
INFRA_DIR="./infrastructure/$ENV"

echo "=== Infrastructure: $ENV ($ACTION) ==="

if [ ! -d "$INFRA_DIR" ]; then
    echo "ERROR: Directory not found: $INFRA_DIR"
    exit 1
fi

cd "$INFRA_DIR"

case "$ACTION" in
    plan)
        terraform init -backend-config="key=${ENV}.tfstate"
        terraform plan -var-file="${ENV}.tfvars" -out=plan.tfplan
        ;;
    apply)
        if [ ! -f plan.tfplan ]; then
            echo "Run plan first: $0 $ENV plan"
            exit 1
        fi
        terraform apply plan.tfplan
        rm plan.tfplan
        ;;
    destroy)
        echo "WARNING: Destroying $ENV infrastructure!"
        read -r -p "Type '$ENV' to confirm: " confirm
        if [ "$confirm" = "$ENV" ]; then
            terraform destroy -var-file="${ENV}.tfvars" -auto-approve
        else
            echo "Aborted."
        fi
        ;;
    output)
        terraform output -json
        ;;
    *)
        echo "Actions: plan, apply, destroy, output"
        ;;
esac`,
				},
				{
					Title: "Infrastructure as Code with Bicep and Terraform",
					Content: `Infrastructure as Code (IaC) enables repeatable, version-controlled Azure deployments. Bicep is Azure-native, while Terraform is multi-cloud.

**Azure Bicep:**
` + "```" + `
Bicep is the Azure-native DSL for ARM templates.

Advantages:
  - Simpler syntax than ARM JSON
  - First-class Azure integration
  - Type safety and intellisense in VS Code
  - No state file needed
  - Automatic dependency resolution
  - Modules for reusability

Basic syntax:
  // main.bicep
  @description('Azure region')
  param location string = resourceGroup().location
  
  @description('Environment name')
  @allowed(['dev', 'staging', 'prod'])
  param environment string
  
  @description('App name')
  param appName string
  
  // Variables
  var namePrefix = '${appName}-${environment}'
  var tags = {
    Environment: environment
    ManagedBy: 'Bicep'
  }
  
  // Storage Account
  resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
    name: '${replace(namePrefix, '-', '')}storage'
    location: location
    tags: tags
    sku: {
      name: 'Standard_LRS'
    }
    kind: 'StorageV2'
    properties: {
      minimumTlsVersion: 'TLS1_2'
      supportsHttpsTrafficOnly: true
      allowBlobPublicAccess: false
    }
  }
  
  // App Service Plan
  resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
    name: '${namePrefix}-plan'
    location: location
    tags: tags
    sku: {
      name: environment == 'prod' ? 'P1V3' : 'B1'
    }
    kind: 'linux'
    properties: {
      reserved: true
    }
  }
  
  // Web App
  resource webApp 'Microsoft.Web/sites@2023-01-01' = {
    name: '${namePrefix}-web'
    location: location
    tags: tags
    properties: {
      serverFarmId: appServicePlan.id
      httpsOnly: true
      siteConfig: {
        linuxFxVersion: 'NODE|18-lts'
        minTlsVersion: '1.2'
        appSettings: [
          {
            name: 'STORAGE_CONNECTION'
            value: storageAccount.properties.primaryEndpoints.blob
          }
        ]
      }
    }
  }
  
  // Outputs
  output webAppUrl string = 'https://${webApp.properties.defaultHostName}'
  output storageEndpoint string = storageAccount.properties.primaryEndpoints.blob

Deploy:
  # Create resource group
  az group create --name myRG --location eastus
  
  # Deploy
  az deployment group create \
    --resource-group myRG \
    --template-file main.bicep \
    --parameters environment=prod appName=myapp
  
  # What-if (preview changes)
  az deployment group what-if \
    --resource-group myRG \
    --template-file main.bicep \
    --parameters environment=prod appName=myapp

Modules:
  // modules/vnet.bicep
  param name string
  param location string
  param addressPrefix string
  
  resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
    name: name
    location: location
    properties: {
      addressSpace: { addressPrefixes: [addressPrefix] }
    }
  }
  output vnetId string = vnet.id
  
  // main.bicep - use module
  module network 'modules/vnet.bicep' = {
    name: 'vnet-deployment'
    params: {
      name: '${namePrefix}-vnet'
      location: location
      addressPrefix: '10.0.0.0/16'
    }
  }
` + "```" + `

**Terraform for Azure:**
` + "```" + `
Terraform uses HCL (HashiCorp Configuration Language) with azurerm provider.

Provider configuration:
  # providers.tf
  terraform {
    required_version = ">= 1.5.0"
    
    required_providers {
      azurerm = {
        source  = "hashicorp/azurerm"
        version = "~> 3.80"
      }
    }
    
    backend "azurerm" {
      resource_group_name  = "terraform-state-rg"
      storage_account_name = "tfstatestorageacct"
      container_name       = "tfstate"
      key                  = "prod.terraform.tfstate"
    }
  }
  
  provider "azurerm" {
    features {}
  }

Resource definitions:
  # main.tf
  resource "azurerm_resource_group" "main" {
    name     = "${var.app_name}-${var.environment}-rg"
    location = var.location
    tags     = local.common_tags
  }
  
  resource "azurerm_virtual_network" "main" {
    name                = "${var.app_name}-vnet"
    resource_group_name = azurerm_resource_group.main.name
    location            = azurerm_resource_group.main.location
    address_space       = ["10.0.0.0/16"]
    tags                = local.common_tags
  }
  
  resource "azurerm_subnet" "app" {
    name                 = "app-subnet"
    resource_group_name  = azurerm_resource_group.main.name
    virtual_network_name = azurerm_virtual_network.main.name
    address_prefixes     = ["10.0.1.0/24"]
    
    delegation {
      name = "webapp"
      service_delegation {
        name = "Microsoft.Web/serverFarms"
      }
    }
  }

Bicep vs Terraform:
  Feature           Bicep              Terraform
  Cloud support     Azure only         Multi-cloud
  State mgmt        Azure (built-in)   Separate state file
  Language          Bicep DSL          HCL
  Learning curve    Lower for Azure    Moderate
  Community         Growing            Very large
  Modularity        Modules            Modules + Registry
  Drift detection   What-if            Plan + state
  
  Choose Bicep if:
  - Azure-only environment
  - Team familiar with ARM templates
  
  Choose Terraform if:
  - Multi-cloud strategy
  - Team already using Terraform
  - Need extensive module ecosystem
` + "```" + ``,
					CodeExamples: `# IaC management scripts

# 1. Bicep deployment wrapper
#!/bin/bash
set -euo pipefail

TEMPLATE="${1:?Usage: $0 <template.bicep> <resource-group> [params-file]}"
RG="${2:?Usage: $0 <template.bicep> <resource-group> [params-file]}"
PARAMS_FILE="${3:-}"

DEPLOYMENT_NAME="deploy-$(date +%Y%m%d-%H%M%S)"

echo "=== Bicep Deployment ==="
echo "Template: $TEMPLATE"
echo "Resource Group: $RG"
echo "Deployment: $DEPLOYMENT_NAME"

# Validate
echo ""
echo "Validating..."
VALIDATE_CMD="az deployment group validate --resource-group $RG --template-file $TEMPLATE --name $DEPLOYMENT_NAME"
if [ -n "$PARAMS_FILE" ]; then
    VALIDATE_CMD="$VALIDATE_CMD --parameters @$PARAMS_FILE"
fi
eval "$VALIDATE_CMD"
echo "Validation passed."

# What-if
echo ""
echo "What-if analysis:"
WHATIF_CMD="az deployment group what-if --resource-group $RG --template-file $TEMPLATE --name $DEPLOYMENT_NAME"
if [ -n "$PARAMS_FILE" ]; then
    WHATIF_CMD="$WHATIF_CMD --parameters @$PARAMS_FILE"
fi
eval "$WHATIF_CMD"

# Confirm
read -r -p "Proceed with deployment? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Deploy
echo ""
echo "Deploying..."
DEPLOY_CMD="az deployment group create --resource-group $RG --template-file $TEMPLATE --name $DEPLOYMENT_NAME"
if [ -n "$PARAMS_FILE" ]; then
    DEPLOY_CMD="$DEPLOY_CMD --parameters @$PARAMS_FILE"
fi
eval "$DEPLOY_CMD"

echo ""
echo "Deployment outputs:"
az deployment group show --resource-group "$RG" --name "$DEPLOYMENT_NAME" \
    --query "properties.outputs" -o json

echo ""
echo "=== Deployment complete ==="

# 2. Terraform Azure state management
#!/bin/bash
echo "=== Terraform State Management ==="

case "${1:-help}" in
    init)
        RG="${2:?Usage: $0 init <rg-name> <storage-account> <location>}"
        STORAGE="${3:?Usage: $0 init <rg-name> <storage-account> <location>}"
        LOCATION="${4:-eastus}"
        
        echo "Creating Terraform state backend..."
        
        az group create --name "$RG" --location "$LOCATION"
        
        az storage account create \
            --resource-group "$RG" --name "$STORAGE" \
            --sku Standard_LRS --kind StorageV2 \
            --min-tls-version TLS1_2 \
            --allow-blob-public-access false
        
        az storage container create \
            --account-name "$STORAGE" --name tfstate
        
        # Enable versioning for state protection
        az storage account blob-service-properties update \
            --account-name "$STORAGE" \
            --enable-versioning true
        
        echo "State backend created:"
        echo "  Resource Group: $RG"
        echo "  Storage Account: $STORAGE"
        echo "  Container: tfstate"
        ;;
    list)
        echo "Terraform state files:"
        STORAGE="${2:?Usage: $0 list <storage-account>}"
        az storage blob list --account-name "$STORAGE" \
            --container-name tfstate --query "[].{name:name, size:properties.contentLength, modified:properties.lastModified}" \
            -o table
        ;;
    *)
        echo "Usage: $0 {init|list} [args]"
        ;;
esac`,
				},
			},
		},
	})
}
