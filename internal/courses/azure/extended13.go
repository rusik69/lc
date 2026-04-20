package azure

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAzureModules([]problems.CourseModule{
		{
			ID:          1259,
			Title:       "Azure DevOps and CI/CD Integration",
			Description: "Master Azure DevOps services including Azure Pipelines, Repos, Artifacts, and integration with GitHub Actions for continuous delivery.",
			Order:       59,
			Lessons: []problems.Lesson{
				{
					Title: "Azure DevOps Services and Pipelines",
					Content: `Azure DevOps provides a complete set of development tools for planning, developing, testing, and delivering software.

**Azure DevOps Services:**
` + "```" + `
Azure DevOps components:
  Azure Boards:     Work item tracking, Kanban, sprints
  Azure Repos:      Git repositories
  Azure Pipelines:  CI/CD pipelines
  Azure Test Plans:  Manual/automated testing
  Azure Artifacts:   Package management (npm, NuGet, Maven, pip)

Azure CLI DevOps extension:
  az extension add --name azure-devops
  az devops configure --defaults organization=https://dev.azure.com/myorg

Create project:
  az devops project create \
    --name MyProject \
    --source-control git \
    --process Agile \
    --visibility private

Azure Repos:
  # Create repo
  az repos create --name my-service --project MyProject
  
  # Clone
  git clone https://dev.azure.com/myorg/MyProject/_git/my-service
  
  # Branch policies
  az repos policy create \
    --repository-id <repo-id> \
    --branch main \
    --policy-type minimumReviewers \
    --settings '{"minimumApproverCount": 2, "creatorVoteCounts": false}'

Azure Artifacts:
  # Create feed
  az artifacts feed create \
    --name my-packages --project MyProject \
    --visibility private
  
  # Upstream sources (proxy public registries)
  # npm, NuGet, Maven, pip, Go
  
  npm config set registry https://pkgs.dev.azure.com/myorg/_packaging/my-packages/npm/registry/
  
  # Retention policies: auto-cleanup old versions
` + "```" + `

**Azure Pipelines (YAML):**
` + "```" + `
Basic pipeline (azure-pipelines.yml):
  trigger:
    branches:
      include:
        - main
        - release/*
    paths:
      include:
        - src/**
      exclude:
        - docs/**
  
  pool:
    vmImage: 'ubuntu-latest'
  
  variables:
    buildConfiguration: 'Release'
    DOCKER_BUILDKIT: 1
  
  stages:
    - stage: Build
      displayName: 'Build and Test'
      jobs:
        - job: BuildJob
          steps:
            - task: UseDotNet@2
              inputs:
                version: '8.0.x'
            
            - script: dotnet build --configuration $(buildConfiguration)
              displayName: 'Build'
            
            - script: dotnet test --configuration $(buildConfiguration) --collect "Code Coverage"
              displayName: 'Test'
            
            - task: PublishTestResults@2
              inputs:
                testResultsFormat: 'VSTest'
                testResultsFiles: '**/*.trx'
            
            - task: PublishBuildArtifacts@1
              inputs:
                pathToPublish: '$(Build.ArtifactStagingDirectory)'
                artifactName: 'drop'
    
    - stage: Deploy_Dev
      displayName: 'Deploy to Dev'
      dependsOn: Build
      condition: succeeded()
      jobs:
        - deployment: DeployDev
          environment: 'dev'
          strategy:
            runOnce:
              deploy:
                steps:
                  - task: AzureWebApp@1
                    inputs:
                      azureSubscription: 'my-service-connection'
                      appName: 'myapp-dev'
                      package: '$(Pipeline.Workspace)/drop/**/*.zip'
    
    - stage: Deploy_Prod
      displayName: 'Deploy to Production'
      dependsOn: Deploy_Dev
      condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
      jobs:
        - deployment: DeployProd
          environment: 'production'
          strategy:
            canary:
              increments: [10, 50]
              deploy:
                steps:
                  - script: echo "Deploying canary"
              on:
                success:
                  steps:
                    - script: echo "Promoting canary"
                failure:
                  steps:
                    - script: echo "Rolling back"

Container build pipeline:
  trigger:
    - main
  
  pool:
    vmImage: 'ubuntu-latest'
  
  variables:
    acrName: myregistry
    imageName: myapp
    tag: $(Build.BuildId)
  
  steps:
    - task: Docker@2
      displayName: 'Build and Push'
      inputs:
        containerRegistry: 'acr-connection'
        repository: '$(imageName)'
        command: 'buildAndPush'
        Dockerfile: '**/Dockerfile'
        tags: |
          $(tag)
          latest
    
    - task: KubernetesManifest@0
      displayName: 'Deploy to AKS'
      inputs:
        action: 'deploy'
        kubernetesServiceConnection: 'aks-connection'
        namespace: 'default'
        manifests: |
          manifests/deployment.yml
          manifests/service.yml
        containers: |
          $(acrName).azurecr.io/$(imageName):$(tag)

Multi-stage with templates:
  # templates/build-template.yml
  parameters:
    - name: project
      type: string
  
  steps:
    - script: dotnet build ${{ parameters.project }}
    - script: dotnet test ${{ parameters.project }}
  
  # azure-pipelines.yml
  stages:
    - stage: Build
      jobs:
        - job: Build
          steps:
            - template: templates/build-template.yml
              parameters:
                project: 'src/MyApp.sln'

Pipeline variables and secrets:
  variables:
    # Regular variable
    - name: environment
      value: 'production'
    
    # Variable group (from Azure DevOps Library)
    - group: my-variable-group
    
    # Secret (from Azure Key Vault)
    - group: keyvault-variables
  
  # Key Vault integration
  variables:
    - group: my-keyvault-group  # linked to Key Vault

Service connections:
  - Azure Resource Manager (service principal)
  - Docker Registry (ACR, Docker Hub)
  - Kubernetes (AKS, kubeconfig)
  - GitHub, Bitbucket
  - SSH, generic
` + "```" + `

**GitHub Actions for Azure:**
` + "```" + `
Deploy to Azure Web App:
  name: Deploy to Azure
  on:
    push:
      branches: [main]
  
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        
        - name: Login to Azure
          uses: azure/login@v2
          with:
            creds: ${{ secrets.AZURE_CREDENTIALS }}
        
        - name: Deploy to Web App
          uses: azure/webapps-deploy@v3
          with:
            app-name: myWebApp
            package: ./dist
        
        - name: Azure logout
          run: az logout

Deploy to AKS:
  name: Deploy to AKS
  on:
    push:
      branches: [main]
  
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        
        - uses: azure/login@v2
          with:
            creds: ${{ secrets.AZURE_CREDENTIALS }}
        
        - uses: azure/aks-set-context@v4
          with:
            resource-group: myRG
            cluster-name: myAKS
        
        - uses: azure/k8s-deploy@v5
          with:
            manifests: manifests/
            images: myacr.azurecr.io/myapp:${{ github.sha }}

Terraform with Azure:
  name: Terraform
  on:
    push:
      branches: [main]
    pull_request:
  
  jobs:
    terraform:
      runs-on: ubuntu-latest
      env:
        ARM_CLIENT_ID: ${{ secrets.ARM_CLIENT_ID }}
        ARM_CLIENT_SECRET: ${{ secrets.ARM_CLIENT_SECRET }}
        ARM_SUBSCRIPTION_ID: ${{ secrets.ARM_SUBSCRIPTION_ID }}
        ARM_TENANT_ID: ${{ secrets.ARM_TENANT_ID }}
      
      steps:
        - uses: actions/checkout@v4
        - uses: hashicorp/setup-terraform@v3
        
        - run: terraform init
        - run: terraform plan -out=tfplan
          if: github.event_name == 'pull_request'
        
        - run: terraform apply -auto-approve
          if: github.ref == 'refs/heads/main'
` + "```" + ``,
					CodeExamples: `# Azure DevOps management scripts

# 1. Pipeline status dashboard
#!/bin/bash
echo "=== Azure Pipelines Status ==="

ORG="https://dev.azure.com/myorg"
PROJECT="MyProject"

# List recent pipeline runs
echo "--- Recent Pipeline Runs ---"
az pipelines runs list \
    --org "$ORG" -p "$PROJECT" \
    --top 10 \
    --query "[].{id:id, pipeline:definition.name, status:status, result:result, branch:sourceBranch}" \
    -o table 2>/dev/null

# Pipeline definitions
echo ""
echo "--- Pipeline Definitions ---"
az pipelines list \
    --org "$ORG" -p "$PROJECT" \
    --query "[].{id:id, name:name, folder:folder}" \
    -o table 2>/dev/null

# Build durations
echo ""
echo "--- Average Build Duration (last 20 runs) ---"
az pipelines runs list \
    --org "$ORG" -p "$PROJECT" \
    --top 20 \
    --query "[?result=='succeeded'].{
        pipeline:definition.name,
        start:startTime, finish:finishTime
    }" -o json 2>/dev/null | jq -r '.[] | "\(.pipeline): \(.start) -> \(.finish)"'

# 2. Environment management
#!/bin/bash
echo "=== Environment Status ==="

# Check environments
for env in dev staging production; do
    echo "--- $env ---"
    
    # App Service
    APP="myapp-$env"
    STATUS=$(az webapp show -n "$APP" -g "rg-$env" --query "state" -o tsv 2>/dev/null)
    URL=$(az webapp show -n "$APP" -g "rg-$env" --query "defaultHostName" -o tsv 2>/dev/null)
    echo "  App Service: $STATUS ($URL)"
    
    # Health check
    if [ -n "$URL" ]; then
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" "https://$URL/health" 2>/dev/null)
        echo "  Health: HTTP $HTTP"
    fi
    
    # Latest deployment
    az webapp deployment list-publishing-credentials \
        -n "$APP" -g "rg-$env" \
        --query "{user:publishingUserName}" -o tsv 2>/dev/null
done

# 3. Release gating script
#!/bin/bash
echo "=== Pre-Deployment Validation ==="

ENVIRONMENT="${1:-staging}"
APP_NAME="myapp-$ENVIRONMENT"
RG="rg-$ENVIRONMENT"

CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    local name="$1"
    local result="$2"
    ((CHECKS_TOTAL++))
    if [ "$result" = "PASS" ]; then
        ((CHECKS_PASSED++))
        echo "  [PASS] $name"
    else
        echo "  [FAIL] $name"
    fi
}

# Check resource group exists
RG_EXISTS=$(az group exists -n "$RG" 2>/dev/null)
check "Resource group exists" "$([ "$RG_EXISTS" = "true" ] && echo PASS || echo FAIL)"

# Check app service running
APP_STATE=$(az webapp show -n "$APP_NAME" -g "$RG" --query "state" -o tsv 2>/dev/null)
check "App Service running" "$([ "$APP_STATE" = "Running" ] && echo PASS || echo FAIL)"

# Check SSL certificate
CERT_EXPIRY=$(az webapp config ssl list -g "$RG" \
    --query "[?name=='$APP_NAME'].expirationDate" -o tsv 2>/dev/null)
if [ -n "$CERT_EXPIRY" ]; then
    DAYS_LEFT=$(( ( $(date -d "$CERT_EXPIRY" +%s 2>/dev/null || echo 0) - $(date +%s) ) / 86400 ))
    check "SSL cert valid (>30 days)" "$([ "$DAYS_LEFT" -gt 30 ] && echo PASS || echo FAIL)"
fi

# Check database connectivity
DB_STATUS=$(az sql db show -s "sql-$ENVIRONMENT" -n "db-$ENVIRONMENT" -g "$RG" \
    --query "status" -o tsv 2>/dev/null)
check "Database online" "$([ "$DB_STATUS" = "Online" ] && echo PASS || echo FAIL)"

echo ""
echo "Results: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
[ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ] && echo "READY FOR DEPLOYMENT" || echo "DEPLOYMENT BLOCKED"`,
				},
				{
					Title: "Azure Infrastructure as Code with Bicep",
					Content: `Bicep is Azure's domain-specific language for deploying Azure resources declaratively with concise syntax.

**Bicep Fundamentals:**
` + "```" + `
Bicep vs ARM:
  ARM JSON:  Verbose, complex, hard to read
  Bicep:     Concise, type-safe, IntelliSense support
  
  Bicep compiles to ARM JSON templates.
  All ARM template capabilities are available.

Install Bicep:
  az bicep install
  az bicep upgrade
  az bicep version

Basic syntax:
  // main.bicep
  
  // Parameters (inputs)
  @description('Environment name')
  @allowed(['dev', 'staging', 'prod'])
  param environment string = 'dev'
  
  @description('Azure region')
  param location string = resourceGroup().location
  
  @minLength(3)
  @maxLength(24)
  param storageAccountName string
  
  @secure()
  param adminPassword string
  
  // Variables (computed values)
  var prefix = 'myapp-${environment}'
  var tags = {
    Environment: environment
    ManagedBy: 'Bicep'
  }
  
  // Resources
  resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
    name: storageAccountName
    location: location
    tags: tags
    kind: 'StorageV2'
    sku: {
      name: environment == 'prod' ? 'Standard_GRS' : 'Standard_LRS'
    }
    properties: {
      minimumTlsVersion: 'TLS1_2'
      allowBlobPublicAccess: false
      supportsHttpsTrafficOnly: true
    }
  }
  
  // Child resources
  resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
    parent: storageAccount
    name: 'default'
    properties: {
      containerDeleteRetentionPolicy: {
        days: 7
        enabled: true
      }
    }
  }
  
  resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
    parent: blobService
    name: 'data'
    properties: {
      publicAccess: 'None'
    }
  }
  
  // Outputs
  output storageId string = storageAccount.id
  output blobEndpoint string = storageAccount.properties.primaryEndpoints.blob

Deploy:
  # Create resource group
  az group create -n myRG -l eastus
  
  # Deploy
  az deployment group create \
    -g myRG \
    --template-file main.bicep \
    --parameters environment=dev storageAccountName=mystg123
  
  # What-if (preview changes)
  az deployment group what-if \
    -g myRG \
    --template-file main.bicep \
    --parameters environment=dev storageAccountName=mystg123
` + "```" + `

**Bicep Modules and Patterns:**
` + "```" + `
Modules (reusable components):
  // modules/appservice.bicep
  @description('App name')
  param appName string
  
  @description('Location')
  param location string
  
  @description('SKU')
  param sku string = 'B1'
  
  resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
    name: '${appName}-plan'
    location: location
    sku: {
      name: sku
    }
    kind: 'linux'
    properties: {
      reserved: true
    }
  }
  
  resource webApp 'Microsoft.Web/sites@2023-01-01' = {
    name: appName
    location: location
    properties: {
      serverFarmId: appServicePlan.id
      httpsOnly: true
      siteConfig: {
        linuxFxVersion: 'DOTNETCORE|8.0'
        minTlsVersion: '1.2'
        ftpsState: 'Disabled'
      }
    }
  }
  
  output appUrl string = 'https://${webApp.properties.defaultHostName}'
  output appId string = webApp.id

  // main.bicep - using module
  param environment string
  param location string = resourceGroup().location
  
  module webApp 'modules/appservice.bicep' = {
    name: 'webAppDeploy'
    params: {
      appName: 'myapp-${environment}'
      location: location
      sku: environment == 'prod' ? 'P1v3' : 'B1'
    }
  }
  
  output url string = webApp.outputs.appUrl

Loops:
  param storageNames array = ['logs', 'data', 'backup']
  
  resource storageAccounts 'Microsoft.Storage/storageAccounts@2023-01-01' = [
    for name in storageNames: {
      name: 'stg${name}${uniqueString(resourceGroup().id)}'
      location: resourceGroup().location
      kind: 'StorageV2'
      sku: { name: 'Standard_LRS' }
    }
  ]
  
  // Loop with index
  resource nics 'Microsoft.Network/networkInterfaces@2023-06-01' = [
    for i in range(0, 3): {
      name: 'nic-${i}'
      location: resourceGroup().location
      properties: {
        ipConfigurations: [{
          name: 'ipconfig1'
          properties: {
            privateIPAllocationMethod: 'Dynamic'
            subnet: { id: subnetId }
          }
        }]
      }
    }
  ]

Conditions:
  param deployRedis bool = false
  
  resource redisCache 'Microsoft.Cache/redis@2023-08-01' = if (deployRedis) {
    name: 'redis-${environment}'
    location: location
    properties: {
      sku: {
        name: 'Basic'
        family: 'C'
        capacity: 0
      }
    }
  }

Existing resources:
  // Reference existing resource
  resource existingVnet 'Microsoft.Network/virtualNetworks@2023-06-01' existing = {
    name: 'my-vnet'
    scope: resourceGroup('networking-rg')
  }
  
  var subnetId = existingVnet.properties.subnets[0].id

Deployment scopes:
  // Resource group (default)
  targetScope = 'resourceGroup'
  
  // Subscription level
  targetScope = 'subscription'
  resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
    name: 'myRG'
    location: 'eastus'
  }
  
  module resources 'main.bicep' = {
    name: 'resourcesDeploy'
    scope: rg
    params: { ... }
  }
  
  // Management group level
  targetScope = 'managementGroup'
  
  // Tenant level
  targetScope = 'tenant'
` + "```" + `

**Terraform for Azure:**
` + "```" + `
Azure Provider:
  terraform {
    required_providers {
      azurerm = {
        source  = "hashicorp/azurerm"
        version = "~> 3.0"
      }
    }
    backend "azurerm" {
      resource_group_name  = "tfstate"
      storage_account_name = "tfstate123"
      container_name       = "tfstate"
      key                  = "terraform.tfstate"
    }
  }
  
  provider "azurerm" {
    features {}
    subscription_id = var.subscription_id
  }

State in Azure Storage:
  # Create storage for state
  az group create -n tfstate -l eastus
  az storage account create -n tfstate123 -g tfstate -l eastus --sku Standard_LRS
  az storage container create -n tfstate --account-name tfstate123

Common resources:
  resource "azurerm_resource_group" "main" {
    name     = "rg-${var.environment}"
    location = var.location
    tags     = var.tags
  }
  
  resource "azurerm_virtual_network" "main" {
    name                = "vnet-${var.environment}"
    location            = azurerm_resource_group.main.location
    resource_group_name = azurerm_resource_group.main.name
    address_space       = ["10.0.0.0/16"]
  }
  
  resource "azurerm_kubernetes_cluster" "main" {
    name                = "aks-${var.environment}"
    location            = azurerm_resource_group.main.location
    resource_group_name = azurerm_resource_group.main.name
    dns_prefix          = "aks-${var.environment}"
    
    default_node_pool {
      name       = "default"
      node_count = var.node_count
      vm_size    = "Standard_D2s_v5"
    }
    
    identity {
      type = "SystemAssigned"
    }
  }
` + "```" + ``,
					CodeExamples: `# Azure IaC management scripts

# 1. Bicep deployment wrapper
#!/bin/bash
set -e

TEMPLATE="${1:-main.bicep}"
PARAMS_FILE="${2:-parameters.json}"
ENVIRONMENT="${3:-dev}"
RG="rg-$ENVIRONMENT"
LOCATION="eastus"

echo "=== Bicep Deployment ==="
echo "Template: $TEMPLATE"
echo "Environment: $ENVIRONMENT"
echo "Resource Group: $RG"

# Validate
echo ""
echo "--- Validating ---"
az deployment group validate \
    -g "$RG" \
    --template-file "$TEMPLATE" \
    --parameters "@$PARAMS_FILE" \
    --parameters environment="$ENVIRONMENT" 2>&1

# What-if
echo ""
echo "--- What-If Preview ---"
az deployment group what-if \
    -g "$RG" \
    --template-file "$TEMPLATE" \
    --parameters "@$PARAMS_FILE" \
    --parameters environment="$ENVIRONMENT"

# Confirm
echo ""
read -p "Proceed with deployment? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "--- Deploying ---"
    az deployment group create \
        -g "$RG" \
        --template-file "$TEMPLATE" \
        --parameters "@$PARAMS_FILE" \
        --parameters environment="$ENVIRONMENT" \
        --name "deploy-$(date +%Y%m%d%H%M%S)"
    
    echo "--- Deployment Complete ---"
    az deployment group show \
        -g "$RG" \
        --name "deploy-$(date +%Y%m%d%H%M%S)" \
        --query "properties.outputs" -o json 2>/dev/null
fi

# 2. Deployment history checker
#!/bin/bash
echo "=== Deployment History ==="

RG="${1:-myRG}"

az deployment group list -g "$RG" \
    --query "[].{
        name:name,
        state:properties.provisioningState,
        timestamp:properties.timestamp,
        duration:properties.duration
    }" -o table | head -20

# Failed deployments
echo ""
echo "--- Failed Deployments ---"
FAILED=$(az deployment group list -g "$RG" \
    --query "[?properties.provisioningState=='Failed'].name" -o tsv)

for deploy in $FAILED; do
    echo "Deployment: $deploy"
    az deployment group show -g "$RG" -n "$deploy" \
        --query "properties.error" -o json 2>/dev/null | jq .
done

# 3. Bicep module registry
#!/bin/bash
echo "=== Bicep Module Registry ==="

ACR_NAME="bicepmodules"

# Create ACR for modules
az acr create -n "$ACR_NAME" -g "shared" --sku Basic 2>/dev/null

# Publish module
publish_module() {
    local module_path="$1"
    local module_name="$2"
    local version="$3"
    
    echo "Publishing $module_name:$version"
    az bicep publish \
        --file "$module_path" \
        --target "br:${ACR_NAME}.azurecr.io/bicep/modules/${module_name}:${version}"
}

# Usage in Bicep:
# module webApp 'br:bicepmodules.azurecr.io/bicep/modules/appservice:v1.0' = {
#   params: { appName: 'myapp', location: location }
# }`,
				},
			},
		},
	})
}
