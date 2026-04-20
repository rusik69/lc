package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		{
			ID:          1454,
			Title:       "Infrastructure as Code Deep Dive",
			Description: "Master Infrastructure as Code with Terraform, Pulumi, and CloudFormation including state management, modules, testing, and drift detection.",
			Order:       54,
			Lessons: []problems.Lesson{
				{
					Title: "Terraform Advanced Patterns",
					Content: `Terraform provides a declarative approach to infrastructure provisioning with advanced patterns for managing complex deployments.

**Terraform State Management:**
` + "```" + `
State backends:
  Local:      terraform.tfstate file
  S3:         AWS S3 bucket + DynamoDB for locking
  GCS:        Google Cloud Storage
  Azure Blob: Azure Storage Account
  Terraform Cloud: HashiCorp managed

S3 backend configuration:
  terraform {
    backend "s3" {
      bucket         = "my-terraform-state"
      key            = "prod/infrastructure.tfstate"
      region         = "us-east-1"
      encrypt        = true
      dynamodb_table = "terraform-locks"
      kms_key_id     = "alias/terraform-state"
    }
  }

State management commands:
  # List resources in state
  terraform state list
  
  # Show specific resource
  terraform state show aws_instance.web
  
  # Move resource (rename)
  terraform state mv aws_instance.old aws_instance.new
  
  # Import existing resource
  terraform import aws_instance.web i-1234567890
  
  # Remove from state (without destroying)
  terraform state rm aws_instance.legacy
  
  # Pull remote state locally
  terraform state pull > state_backup.json

State isolation:
  Per-environment:
    environments/
    ├── dev/
    │   ├── main.tf
    │   └── backend.tf  (key = "dev/terraform.tfstate")
    ├── staging/
    │   ├── main.tf
    │   └── backend.tf  (key = "staging/terraform.tfstate")
    └── prod/
        ├── main.tf
        └── backend.tf  (key = "prod/terraform.tfstate")
  
  Workspaces (alternative):
    terraform workspace new dev
    terraform workspace new prod
    terraform workspace select dev
    
    # Access in code
    resource "aws_instance" "web" {
      tags = {
        Environment = terraform.workspace
      }
    }
  
  Remote state data source:
    data "terraform_remote_state" "networking" {
      backend = "s3"
      config = {
        bucket = "my-terraform-state"
        key    = "networking/terraform.tfstate"
        region = "us-east-1"
      }
    }
    
    # Use outputs from other state
    resource "aws_instance" "web" {
      subnet_id = data.terraform_remote_state.networking.outputs.subnet_id
    }
` + "```" + `

**Terraform Modules:**
` + "```" + `
Module structure:
  modules/
  └── vpc/
      ├── main.tf
      ├── variables.tf
      ├── outputs.tf
      ├── versions.tf
      └── README.md

Module definition:
  # modules/vpc/variables.tf
  variable "name" {
    type        = string
    description = "VPC name"
  }
  
  variable "cidr_block" {
    type    = string
    default = "10.0.0.0/16"
  }
  
  variable "availability_zones" {
    type    = list(string)
    default = ["us-east-1a", "us-east-1b"]
  }
  
  # modules/vpc/main.tf
  resource "aws_vpc" "main" {
    cidr_block           = var.cidr_block
    enable_dns_hostnames = true
    enable_dns_support   = true
    
    tags = {
      Name = var.name
    }
  }
  
  resource "aws_subnet" "public" {
    count             = length(var.availability_zones)
    vpc_id            = aws_vpc.main.id
    cidr_block        = cidrsubnet(var.cidr_block, 8, count.index)
    availability_zone = var.availability_zones[count.index]
    
    tags = {
      Name = "${var.name}-public-${count.index}"
    }
  }
  
  # modules/vpc/outputs.tf
  output "vpc_id" {
    value = aws_vpc.main.id
  }
  
  output "subnet_ids" {
    value = aws_subnet.public[*].id
  }

Module usage:
  module "vpc" {
    source = "./modules/vpc"
    
    name               = "production"
    cidr_block         = "10.0.0.0/16"
    availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  }
  
  module "vpc_staging" {
    source = "./modules/vpc"
    
    name       = "staging"
    cidr_block = "10.1.0.0/16"
  }

Module sources:
  # Local path
  source = "./modules/vpc"
  
  # GitHub
  source = "github.com/myorg/terraform-modules//vpc?ref=v1.0.0"
  
  # Terraform Registry
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  # S3
  source = "s3::https://s3.amazonaws.com/my-bucket/modules/vpc.zip"
  
  # Git
  source = "git::https://git.example.com/modules.git//vpc?ref=v1.0.0"

for_each with modules:
  locals {
    environments = {
      dev = {
        cidr_block = "10.0.0.0/16"
        instance_type = "t3.small"
      }
      staging = {
        cidr_block = "10.1.0.0/16"
        instance_type = "t3.medium"
      }
      prod = {
        cidr_block = "10.2.0.0/16"
        instance_type = "t3.large"
      }
    }
  }
  
  module "environment" {
    for_each = local.environments
    source   = "./modules/environment"
    
    name          = each.key
    cidr_block    = each.value.cidr_block
    instance_type = each.value.instance_type
  }
` + "```" + `

**Terraform Testing and Validation:**
` + "```" + `
Validation:
  variable "environment" {
    type = string
    validation {
      condition     = contains(["dev", "staging", "prod"], var.environment)
      error_message = "Environment must be dev, staging, or prod."
    }
  }
  
  variable "instance_type" {
    type = string
    validation {
      condition     = can(regex("^t3\\.", var.instance_type))
      error_message = "Only t3 instances allowed."
    }
  }

Preconditions and postconditions:
  resource "aws_instance" "web" {
    instance_type = var.instance_type
    ami           = var.ami_id
    
    lifecycle {
      precondition {
        condition     = data.aws_ami.selected.architecture == "x86_64"
        error_message = "AMI must be x86_64 architecture."
      }
      postcondition {
        condition     = self.public_ip != ""
        error_message = "Instance must have a public IP."
      }
    }
  }

Testing with Terratest (Go):
  func TestVPC(t *testing.T) {
    terraformOptions := &terraform.Options{
      TerraformDir: "../modules/vpc",
      Vars: map[string]interface{}{
        "name":       "test",
        "cidr_block": "10.0.0.0/16",
      },
    }
    
    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)
    
    vpcId := terraform.Output(t, terraformOptions, "vpc_id")
    assert.NotEmpty(t, vpcId)
    
    vpc := aws.GetVpcById(t, vpcId, "us-east-1")
    assert.Equal(t, "10.0.0.0/16", vpc.CidrBlock)
  }

terraform test (built-in, HCL):
  # tests/vpc_test.tftest.hcl
  run "create_vpc" {
    command = apply
    
    variables {
      name       = "test"
      cidr_block = "10.0.0.0/16"
    }
    
    assert {
      condition     = aws_vpc.main.cidr_block == "10.0.0.0/16"
      error_message = "VPC CIDR block mismatch"
    }
  }

Plan analysis:
  # Generate plan
  terraform plan -out=tfplan
  
  # Convert to JSON for analysis
  terraform show -json tfplan > plan.json
  
  # Check for destructive changes
  jq '.resource_changes[] | select(.change.actions | contains(["delete"]))' plan.json
  
  # OPA policy check
  conftest test plan.json -p policy/

Drift detection:
  # Detect drift
  terraform plan -detailed-exitcode
  # Exit code 0: no changes
  # Exit code 1: error
  # Exit code 2: changes detected (drift)
  
  # Automated drift detection (cron)
  #!/bin/bash
  terraform plan -detailed-exitcode -out=drift.plan
  if [ $? -eq 2 ]; then
    echo "DRIFT DETECTED"
    terraform show drift.plan
    # Send notification
  fi
` + "```" + ``,
					CodeExamples: `# Infrastructure as Code management scripts

# 1. Terraform workspace manager
#!/bin/bash
echo "=== Terraform Workspace Manager ==="

TF_DIR="${1:-.}"
cd "$TF_DIR" || exit 1

echo "Directory: $TF_DIR"
echo ""

# Current workspace
CURRENT=$(terraform workspace show 2>/dev/null)
echo "Current workspace: $CURRENT"

# List workspaces
echo ""
echo "--- Workspaces ---"
terraform workspace list 2>/dev/null

# State resources per workspace
echo ""
echo "--- Resources per Workspace ---"
for ws in $(terraform workspace list 2>/dev/null | tr -d '* '); do
    terraform workspace select "$ws" > /dev/null 2>&1
    COUNT=$(terraform state list 2>/dev/null | wc -l | tr -d ' ')
    echo "  $ws: $COUNT resources"
done
terraform workspace select "$CURRENT" > /dev/null 2>&1

# 2. Terraform drift detector
#!/bin/bash
echo "=== Drift Detection ==="

TF_DIR="${1:-.}"
cd "$TF_DIR" || exit 1

terraform init -backend=true > /dev/null 2>&1

terraform plan -detailed-exitcode -no-color 2>&1 | tee /tmp/tf-plan.txt
EXIT_CODE=${PIPESTATUS[0]}

case $EXIT_CODE in
    0) echo "NO DRIFT: Infrastructure matches state" ;;
    1) echo "ERROR: Plan failed" ;;
    2)
        echo "DRIFT DETECTED!"
        echo ""
        echo "--- Changes ---"
        grep -E "^  [#+~-]" /tmp/tf-plan.txt
        ;;
esac

# 3. Module documentation generator
#!/bin/bash
echo "=== Module Documentation ==="

MODULE_DIR="${1:-.}"

echo "# $(basename "$MODULE_DIR") Module"
echo ""

# Variables
echo "## Variables"
echo ""
if [ -f "$MODULE_DIR/variables.tf" ]; then
    grep -A3 'variable "' "$MODULE_DIR/variables.tf" | \
        sed -n 's/variable "\(.*\)" {/| \1/p; s/.*description = "\(.*\)"/\1 |/p; s/.*type.*= \(.*\)/\1 |/p; s/.*default.*= \(.*\)/\1 |/p'
fi

# Outputs
echo ""
echo "## Outputs"
echo ""
if [ -f "$MODULE_DIR/outputs.tf" ]; then
    grep -A2 'output "' "$MODULE_DIR/outputs.tf" | \
        sed -n 's/output "\(.*\)" {/- **\1**/p; s/.*description = "\(.*\)"/  \1/p'
fi`,
				},
				{
					Title: "Pulumi and Modern IaC Approaches",
					Content: `Pulumi uses general-purpose programming languages for infrastructure, enabling type safety, testing, and code reuse.

**Pulumi Fundamentals:**
` + "```" + `
Pulumi vs Terraform:
  Terraform: HCL (domain-specific language)
  Pulumi:    TypeScript, Python, Go, C#, Java, YAML
  
  Advantages of Pulumi:
    - Full programming language features
    - Type safety and IDE support
    - Existing testing frameworks
    - Familiar to developers
    - Loops, conditionals, functions native
    - Package managers (npm, pip, go mod)
  
  Advantages of Terraform:
    - Larger ecosystem
    - More community modules
    - Simpler for infrastructure teams
    - HCL prevents over-engineering

Pulumi CLI:
  # New project
  pulumi new aws-typescript
  pulumi new gcp-python
  pulumi new azure-go
  
  # Deploy
  pulumi up
  
  # Preview changes
  pulumi preview
  
  # Destroy
  pulumi destroy
  
  # Stack management
  pulumi stack init dev
  pulumi stack select prod
  pulumi stack ls

Pulumi TypeScript example:
  import * as pulumi from "@pulumi/pulumi";
  import * as aws from "@pulumi/aws";
  
  const config = new pulumi.Config();
  const environment = config.require("environment");
  
  // VPC
  const vpc = new aws.ec2.Vpc("main", {
    cidrBlock: "10.0.0.0/16",
    enableDnsHostnames: true,
    tags: { Name: "main-vpc", Environment: environment },
  });
  
  // Subnets with loop
  const azs = ["us-east-1a", "us-east-1b", "us-east-1c"];
  const subnets = azs.map((az, i) =>
    new aws.ec2.Subnet("subnet-" + az, {
      vpcId: vpc.id,
      cidrBlock: "10.0." + i + ".0/24",
      availabilityZone: az,
      tags: { Name: "subnet-" + az },
    })
  );
  
  // EKS cluster
  const cluster = new aws.eks.Cluster("eks", {
    vpcConfig: {
      subnetIds: subnets.map(s => s.id),
    },
    version: "1.29",
  });
  
  // Exports
  export const vpcId = vpc.id;
  export const clusterName = cluster.name;
  export const kubeconfig = cluster.kubeconfigs;

Pulumi Go example:
  package main
  
  import (
    "github.com/pulumi/pulumi-aws/sdk/v6/go/aws/ec2"
    "github.com/pulumi/pulumi/sdk/v3/go/pulumi"
  )
  
  func main() {
    pulumi.Run(func(ctx *pulumi.Context) error {
      vpc, err := ec2.NewVpc(ctx, "main", &ec2.VpcArgs{
        CidrBlock:          pulumi.String("10.0.0.0/16"),
        EnableDnsHostnames: pulumi.Bool(true),
      })
      if err != nil {
        return err
      }
      
      ctx.Export("vpcId", vpc.ID())
      return nil
    })
  }

Pulumi Python example:
  import pulumi
  import pulumi_aws as aws
  
  config = pulumi.Config()
  env = config.require("environment")
  
  vpc = aws.ec2.Vpc("main",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    tags={"Environment": env}
  )
  
  pulumi.export("vpc_id", vpc.id)

State management:
  Pulumi Cloud:     Managed state backend (default)
  Self-managed:     S3, Azure Blob, GCS
  Local:            File system
  
  # Use S3 backend
  pulumi login s3://my-pulumi-state
  
  # Use local
  pulumi login --local
` + "```" + `

**CrossPlane and Kubernetes-Native IaC:**
` + "```" + `
CrossPlane:
  Kubernetes-native infrastructure provisioning.
  
  Concept:
    - Extend Kubernetes API with cloud resources
    - Manage cloud infra with kubectl
    - Composition and abstraction
    - GitOps compatible
  
  Provider:
    apiVersion: pkg.crossplane.io/v1
    kind: Provider
    metadata:
      name: provider-aws
    spec:
      package: xpkg.upbound.io/upbound/provider-aws:v0.40.0
  
  Managed Resource:
    apiVersion: s3.aws.upbound.io/v1beta1
    kind: Bucket
    metadata:
      name: my-bucket
    spec:
      forProvider:
        region: us-east-1
      providerConfigRef:
        name: default
  
  Composition:
    Define reusable infrastructure patterns as XRDs.
    
    apiVersion: apiextensions.crossplane.io/v1
    kind: CompositeResourceDefinition
    metadata:
      name: xdatabases.example.com
    spec:
      group: example.com
      names:
        kind: XDatabase
        plural: xdatabases
      versions:
        - name: v1alpha1
          served: true
          schema:
            openAPIV3Schema:
              type: object
              properties:
                spec:
                  type: object
                  properties:
                    size:
                      type: string
                      enum: [small, medium, large]

CDK for Terraform (CDKTF):
  TypeScript/Python/Go/Java → generates HCL/JSON.
  
  import { Construct } from "constructs";
  import { App, TerraformStack } from "cdktf";
  import { AwsProvider } from "@cdktf/provider-aws/lib/provider";
  import { S3Bucket } from "@cdktf/provider-aws/lib/s3-bucket";
  
  class MyStack extends TerraformStack {
    constructor(scope: Construct, id: string) {
      super(scope, id);
      
      new AwsProvider(this, "AWS", { region: "us-east-1" });
      
      new S3Bucket(this, "bucket", {
        bucket: "my-cdktf-bucket",
        tags: { ManagedBy: "CDKTF" },
      });
    }
  }
  
  const app = new App();
  new MyStack(app, "my-stack");
  app.synth();

IaC comparison:
  Tool          Language    State        Learning Curve
  Terraform     HCL        Remote       Medium
  Pulumi        TS/Py/Go   Cloud/S3     Low (developers)
  CloudForm.    JSON/YAML  AWS-managed  Medium
  Bicep         Bicep      Azure-mngd   Low
  CrossPlane    YAML (K8s) etcd         High
  CDK           TS/Py      CloudForm.   Medium
  CDKTF         TS/Py      Terraform    Medium
` + "```" + ``,
					CodeExamples: `# IaC management scripts

# 1. Terraform project scanner
#!/bin/bash
echo "=== Terraform Projects Scanner ==="

find . -name "*.tf" -not -path "*/.terraform/*" | \
    xargs dirname 2>/dev/null | sort -u | while read -r dir; do
    
    echo "Project: $dir"
    
    # Check provider
    PROVIDERS=$(grep -h 'required_providers' "$dir"/*.tf 2>/dev/null | head -1)
    if [ -n "$PROVIDERS" ]; then
        grep -h 'source.*=' "$dir"/*.tf 2>/dev/null | head -3 | sed 's/^/  /'
    fi
    
    # Count resources
    RESOURCES=$(grep -ch '^resource ' "$dir"/*.tf 2>/dev/null | awk '{s+=$1}END{print s}')
    DATA=$(grep -ch '^data ' "$dir"/*.tf 2>/dev/null | awk '{s+=$1}END{print s}')
    MODULES=$(grep -ch '^module ' "$dir"/*.tf 2>/dev/null | awk '{s+=$1}END{print s}')
    echo "  Resources: $RESOURCES, Data: $DATA, Modules: $MODULES"
    
    # Check backend
    BACKEND=$(grep -l 'backend "' "$dir"/*.tf 2>/dev/null | head -1)
    if [ -n "$BACKEND" ]; then
        BACKEND_TYPE=$(grep 'backend "' "$BACKEND" | sed 's/.*backend "\(.*\)".*/\1/')
        echo "  Backend: $BACKEND_TYPE"
    else
        echo "  Backend: local"
    fi
    echo ""
done

# 2. IaC security scanner
#!/bin/bash
echo "=== IaC Security Scan ==="

# Check for hardcoded secrets
echo "--- Potential Secrets ---"
grep -rn "password\|secret\|api_key\|access_key" \
    --include="*.tf" --include="*.yaml" --include="*.yml" \
    --exclude-dir=".terraform" . 2>/dev/null | \
    grep -v "variable\|description\|#\|sensitive\|secretref" | head -10

# Check for public resources
echo ""
echo "--- Potentially Public Resources ---"
grep -rn "publicly_accessible.*true\|public_access.*true\|ingress.*0.0.0.0" \
    --include="*.tf" . 2>/dev/null | head -10

# Run tfsec if available
if command -v tfsec &>/dev/null; then
    echo ""
    echo "--- tfsec Results ---"
    tfsec . --format simple 2>/dev/null | head -30
fi

# Run checkov if available
if command -v checkov &>/dev/null; then
    echo ""
    echo "--- checkov Results ---"
    checkov -d . --compact --quiet 2>/dev/null | head -30
fi

# 3. Pulumi stack reporter
#!/bin/bash
echo "=== Pulumi Stacks ==="

# List stacks
pulumi stack ls 2>/dev/null | head -20

echo ""
echo "--- Current Stack ---"
CURRENT=$(pulumi stack --show-name 2>/dev/null)
echo "Stack: $CURRENT"

echo ""
echo "--- Stack Outputs ---"
pulumi stack output 2>/dev/null

echo ""
echo "--- Resource Count ---"
pulumi stack export 2>/dev/null | jq '.deployment.resources | length'`,
				},
			},
		},
	})
}
