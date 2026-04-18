package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2115,
			Title:       "AWS Well-Architected Framework",
			Description: "Learn the AWS Well-Architected Framework's six pillars: operational excellence, security, reliability, performance efficiency, cost optimization, and sustainability.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "The Six Pillars Overview",
					Content: `The AWS Well-Architected Framework provides a consistent approach for evaluating architectures and implementing designs that scale over time. It's based on six pillars.

**1. Operational Excellence:**
- Run and monitor systems to deliver business value
- Continuously improve processes and procedures
- Key practices: Infrastructure as Code, small frequent changes, anticipate failure
- Tools: CloudFormation, CloudWatch, Systems Manager, Config

**2. Security:**
- Protect information, systems, and assets
- Identity and access management, detection, infrastructure protection, data protection
- Key practices: Least privilege, encrypt everywhere, automate security
- Tools: IAM, KMS, GuardDuty, SecurityHub, WAF, Shield

**3. Reliability:**
- Ensure workloads perform correctly and consistently
- Recover from failures, meet demand
- Key practices: Auto-recovery, horizontal scaling, testing recovery
- Tools: Auto Scaling, Route 53 health checks, Multi-AZ/Multi-Region

**4. Performance Efficiency:**
- Use compute resources efficiently to meet requirements
- Maintain efficiency as demand changes and technologies evolve
- Key practices: Right-sizing, caching, serverless where appropriate
- Tools: CloudWatch metrics, Compute Optimizer, Lambda, CloudFront

**5. Cost Optimization:**
- Avoid unnecessary costs
- Understanding spending, selecting right resources, managing demand
- Key practices: Reserved Instances, Spot instances, right-sizing, lifecycle policies
- Tools: Cost Explorer, Budgets, Savings Plans, Trusted Advisor

**6. Sustainability:**
- Minimize environmental impact
- Key practices: Region selection, efficient resource utilization, managed services
- Tools: Customer Carbon Footprint Tool

**Architecture Review:**
Regularly review workloads against the Well-Architected Framework using:
- AWS Well-Architected Tool (free in console)
- Identify high-risk issues
- Create improvement plans
- Track progress over time

**Common Anti-Patterns:**
- Over-provisioning (paying for unused capacity)
- Single AZ deployment (no fault tolerance)
- No encryption (security risk)
- No monitoring/alerting (blind to issues)
- Manual deployments (error-prone, slow)
- No backup/recovery testing (false sense of security)`,
					CodeExamples: `# Well-Architected Infrastructure Example (Terraform)

# Pillar 1: Operational Excellence — Infrastructure as Code
resource "aws_launch_template" "app" {
  name_prefix   = "app-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = "t3.medium"

  user_data = base64encode(<<-EOF
    #!/bin/bash
    yum update -y
    amazon-linux-extras install docker -y
    systemctl start docker
    docker pull myapp:latest
    docker run -d -p 80:8080 myapp:latest
  EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = { Name = "app-server" }
  }
}

# Pillar 3: Reliability — Auto Scaling across AZs
resource "aws_autoscaling_group" "app" {
  desired_capacity    = 2
  max_size           = 10
  min_size           = 2
  vpc_zone_identifier = [
    aws_subnet.private_a.id,
    aws_subnet.private_b.id,
    aws_subnet.private_c.id,  # 3 AZs for reliability
  ]

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }

  health_check_type         = "ELB"
  health_check_grace_period = 300
}

# Pillar 4: Performance — CloudFront CDN
resource "aws_cloudfront_distribution" "cdn" {
  origin {
    domain_name = aws_lb.app.dns_name
    origin_id   = "app-alb"
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "app-alb"
    compress         = true  # Enable compression

    forwarded_values {
      query_string = false
    }
    viewer_protocol_policy = "redirect-to-https"
  }
}

# Pillar 5: Cost — Spot instances for non-critical workloads
resource "aws_launch_template" "batch" {
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.05"
    }
  }
}

# Pillar 2: Security — Encryption at rest
resource "aws_db_instance" "main" {
  engine               = "postgres"
  instance_class       = "db.r6g.large"
  storage_encrypted    = true
  kms_key_id          = aws_kms_key.db.arn
  multi_az            = true
  backup_retention_period = 7
}`,
				},
				{
					Title: "AWS Cost Optimization Strategies",
					Content: `Cost optimization is one of the most impactful skills for any cloud engineer. AWS spending can easily spiral without proper controls.

**Compute Cost Optimization:**

**1. Right-Sizing:**
- Use Compute Optimizer recommendations
- Most instances are over-provisioned by 40-60%
- Check CPU utilization — below 20% average means downsize
- Consider burstable instances (t3/t4g) for variable workloads

**2. Pricing Models:**

| Model | Savings | Commitment | Best For |
|-------|---------|-----------|----------|
| On-Demand | 0% | None | Testing, unpredictable workloads |
| Reserved (1yr) | ~40% | 1 year | Steady-state production |
| Reserved (3yr) | ~60% | 3 years | Long-term, known capacity |
| Savings Plans | ~40-60% | 1-3 years | Flexible (any instance type) |
| Spot Instances | ~70-90% | None (can be interrupted) | Batch, CI/CD, fault-tolerant |

**3. Graviton (ARM) Instances:**
- 20-40% cheaper than x86 equivalents
- Better performance per dollar
- Go, Python, Node.js run natively on ARM
- Types: t4g, m7g, c7g, r7g

**Storage Cost Optimization:**

**S3 Lifecycle Policies:**
- Standard → Infrequent Access (30 days) → Glacier (90 days) → Delete (365 days)
- Can save 60-80% on storage costs

**EBS Optimization:**
- Delete unattached volumes (common waste)
- Use gp3 instead of gp2 (20% cheaper, better performance)
- Snapshot lifecycle policies

**Database Cost Optimization:**
- Aurora Serverless for variable-load databases
- Read replicas for read-heavy workloads
- Reserved instances for production databases
- DynamoDB on-demand vs provisioned capacity

**Serverless for Cost Optimization:**
- Lambda: Pay per invocation (free tier: 1M requests/month)
- API Gateway: Pay per request
- DynamoDB on-demand: Pay per read/write
- Great for: Low/variable traffic, event-driven workloads

**Monitoring and Alerting:**
- Set AWS Budgets with alerts
- Use Cost Explorer for trend analysis
- Tag all resources (team, environment, project)
- Review monthly with Cost Anomaly Detection
- Use Trusted Advisor for specific recommendations

**Quick Wins (do these first):**
1. Delete unused EBS volumes and Elastic IPs
2. Stop/terminate unused EC2 instances
3. Enable S3 Lifecycle policies
4. Switch gp2 to gp3 volumes
5. Review and delete old snapshots
6. Use Graviton instances where possible`,
					CodeExamples: `# Cost monitoring with AWS CLI

# Check monthly spend
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-02-01 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Find unattached EBS volumes (waste!)
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --query 'Volumes[*].[VolumeId,Size,CreateTime]' \
  --output table

# Find unused Elastic IPs
aws ec2 describe-addresses \
  --query 'Addresses[?AssociationId==null].[PublicIp,AllocationId]'

# Set up budget alert (Terraform)
resource "aws_budgets_budget" "monthly" {
  name         = "monthly-budget"
  budget_type  = "COST"
  limit_amount = "5000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type           = "PERCENTAGE"
    notification_type        = "ACTUAL"
    subscriber_email_addresses = ["team@company.com"]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type           = "PERCENTAGE"
    notification_type        = "FORECASTED"
    subscriber_email_addresses = ["team@company.com"]
  }
}

# S3 lifecycle policy for cost savings
resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "log-lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"  # ~45% cheaper
    }

    transition {
      days          = 90
      storage_class = "GLACIER"      # ~80% cheaper
    }

    expiration {
      days = 365  # Delete after 1 year
    }
  }
}

# Auto Scaling with mixed instances (cost + reliability)
resource "aws_autoscaling_group" "mixed" {
  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 1  # 1 on-demand
      on_demand_percentage_above_base_capacity = 25 # 25% on-demand
      spot_allocation_strategy                 = "capacity-optimized"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.app.id
      }

      override {
        instance_type = "t3.medium"
      }
      override {
        instance_type = "t3a.medium"  # AMD, cheaper
      }
      override {
        instance_type = "t4g.medium"  # Graviton, cheapest
      }
    }
  }
}`,
				},
				{
					Title: "AWS Serverless Architecture",
					Content: `Serverless computing lets you build applications without managing servers. AWS Lambda is the core, but the ecosystem includes many managed services.

**The Serverless Stack:**

| Service | Purpose | Pricing |
|---------|---------|---------|
| Lambda | Compute (functions) | $0.20/1M requests + duration |
| API Gateway | HTTP endpoints | $3.50/1M REST requests |
| DynamoDB | NoSQL database | Pay per read/write unit |
| S3 | Object storage | $0.023/GB/month |
| SQS | Message queue | $0.40/1M requests |
| SNS | Notifications | $0.50/1M requests |
| Step Functions | Orchestration | $0.025/1K state transitions |
| EventBridge | Event bus | $1.00/1M events |
| Cognito | Authentication | Free up to 50K MAUs |

**Lambda Best Practices:**

**1. Cold Starts:**
- First invocation after idle period takes longer (100ms-10s)
- Mitigation: Provisioned concurrency, keep-warm pings, SnapStart (Java)
- ARM (Graviton) Lambda has shorter cold starts

**2. Function Design:**
- Keep functions small and focused (SRP)
- Separate business logic from handler
- Use layers for shared code/dependencies
- Set appropriate memory (CPU scales with memory)

**3. Environment Variables:**
- Connection strings, API keys, feature flags
- Use SSM Parameter Store or Secrets Manager for sensitive values
- Never hardcode secrets

**Common Serverless Patterns:**

**1. API + Lambda + DynamoDB:**
` + "```" + `
Client → API Gateway → Lambda → DynamoDB
` + "```" + `
Simple CRUD API. Great for: MVPs, microservices, low-traffic APIs.

**2. Event Processing:**
` + "```" + `
S3 Upload → Lambda → Process → Store Result
SQS Queue → Lambda → Process → DynamoDB
EventBridge Rule → Lambda → Send Notification
` + "```" + `

**3. Fan-out:**
` + "```" + `
SNS Topic → Lambda 1 (Email)
          → Lambda 2 (SMS)
          → Lambda 3 (Slack)
          → SQS Queue → Lambda 4 (Heavy processing)
` + "```" + `

**4. Step Functions (Orchestration):**
` + "```" + `
Start → Validate → Process Payment → Reserve Inventory → Ship → Notify
                    ↓ (fail)
                  Refund → Notify Error
` + "```" + `

**When NOT to Use Serverless:**
- Long-running processes (> 15 minutes)
- High-throughput, low-latency requirements (cold starts)
- WebSocket connections (API Gateway has limited support)
- GPU workloads
- When you need full OS control

**Cost at Scale:**
Serverless is cheap at low scale but can be expensive at high volume. Break-even point: ~1-10M requests/month vs always-on EC2/Fargate. Calculate your specific case.`,
					CodeExamples: `# Lambda function (Python)
import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

def handler(event, context):
    """API Gateway Lambda handler."""
    method = event['httpMethod']
    
    if method == 'GET':
        item_id = event['pathParameters']['id']
        response = table.get_item(Key={'id': item_id})
        
        if 'Item' not in response:
            return {'statusCode': 404, 'body': json.dumps({'error': 'not found'})}
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(response['Item'])
        }
    
    elif method == 'POST':
        body = json.loads(event['body'])
        table.put_item(Item=body)
        
        return {
            'statusCode': 201,
            'body': json.dumps({'message': 'created'})
        }

# Serverless Framework (serverless.yml)
service: my-api

provider:
  name: aws
  runtime: python3.12
  architecture: arm64  # Graviton — cheaper + faster
  environment:
    TABLE_NAME: !Ref ItemsTable

functions:
  api:
    handler: handler.handler
    events:
      - httpApi:
          path: /items/{id}
          method: get
      - httpApi:
          path: /items
          method: post

resources:
  Resources:
    ItemsTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: items
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH

# Step Functions (state machine definition)
{
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:validate",
      "Next": "ProcessPayment",
      "Catch": [{
        "ErrorEquals": ["ValidationError"],
        "Next": "OrderFailed"
      }]
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:payment",
      "Next": "ReserveInventory",
      "Catch": [{
        "ErrorEquals": ["PaymentFailed"],
        "Next": "OrderFailed"
      }]
    },
    "ReserveInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:inventory",
      "Next": "OrderComplete"
    },
    "OrderComplete": {
      "Type": "Succeed"
    },
    "OrderFailed": {
      "Type": "Fail"
    }
  }
}`,
				},
			},
		},
	})
}
