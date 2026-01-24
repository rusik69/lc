package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          320,
			Title:       "Advanced VPC Networking",
			Description: "Advanced VPC topics: VPC peering, VPN, Direct Connect, Transit Gateway, and network architecture patterns.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Networking: Peering vs Transit Gateway",
					Content: `As your AWS footprint grows, connecting VPCs becomes complex. You have two main options:

**1. VPC Peering (The "Direct Cable")**
*   **What is it:** A direct, 1:1 network connection between two VPCs.
*   **Pros:** Lowest latency, no extra "hop", supports Security Group referencing.
*   **Cons:** Non-transitive (A<->B and B<->C does NOT mean A<->C). Mesh topology gets messy (N^2 connections).

**2. AWS Transit Gateway (The "Cloud Router")**
*   **What is it:** A central hub that connects VPCs and on-premises networks.
*   **Pros:** Hub-and-spoke topology (cleaner), transitive routing, centralized control.
*   **Cons:** Extra hop (slightly higher latency), higher cost (processing fees).

**Decision Matrix:**
*   Connecting 2-5 VPCs? -> **VPC Peering** (Simple, cheap).
*   Connecting 100+ VPCs? -> **Transit Gateway** (Manageable).
*   Need transitive routing (VPN -> VPC A -> VPC B)? -> **Transit Gateway**.
*   Need absolute lowest latency/highest bandwidth? -> **VPC Peering**.

**Visual:**
` + "```" + `
[ VPC A ] <--- Peering ---> [ VPC B ]  (Simple)

      [ VPC A ]
          |
[ VPN ]--[ Transit Gateway ]--[ VPC B ] (Hub-and-Spoke)
          |
      [ VPC C ]
` + "```" + `

**PrivateLink (VPC Endpoints):**
Don't forget PrivateLink for securely accessing AWS services (S3, DynamoDB) or SaaS without exposing traffic to the public internet.`,
					CodeExamples: `# Create VPC Peering
aws ec2 create-vpc-peering-connection --vpc-id vpc-1 --peer-vpc-id vpc-2

# Create Transit Gateway
aws ec2 create-transit-gateway --description "Main TGW"

# Route Table: Send traffic to TGW
aws ec2 create-route --route-table-id rtb-1 --destination-cidr-block 10.0.0.0/8 --transit-gateway-id tgw-1`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          321,
			Title:       "AWS Security Best Practices",
			Description: "Advanced security: encryption, secrets management, WAF, Shield, GuardDuty, and security monitoring.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Security Best Practices: IAM Masterclass",
					Content: `IAM is the most critical service in AWS. If you mess up IAM, you get hacked. Period.

**1. The "Whose" and "What"**
*   **Principal:** The entity (User, Role, App) trying to do something.
*   **Action:** The verb (s3:ListBuckets, ec2:StartInstances).
*   **Resource:** The object (arn:aws:s3:::my-photos).
*   **Condition:** The constraints (IpAddress, Date).

**2. IAM Policy Visualized:**
` + "```" + `
{
  "Effect": "Allow",          <-- Green Light
  "Principal": { "AWS": "arn:aws:iam::123:user/Bob" },
  "Action": "s3:GetObject",   <-- Can READ
  "Resource": "arn:aws:s3:::my-bucket/*",
  "Condition": {              <-- BUT ONLY IF...
     "IpAddress": { "aws:SourceIp": "1.2.3.4/32" }
  }
}
` + "```" + `

**3. Best Practices (The Holy Grail):**
*   **Root Account:** Lock it away. Enable MFA. Never use it for daily tasks.
*   **Least Privilege:** Give *only* the permissions needed. Start with 0 and add.
*   **Roles > Users:** Use Roles for applications (EC2, Lambda). Never embed Access Keys in code.
*   **Secrets Manager:** Store DB passwords here, not in environment variables.

**4. Encryption everywhere:**
*   **At Rest:** Click the "Enable Encryption" checkbox (S3, EBS, RDS). It uses KMS.
*   **In Transit:** Use HTTPS (TLS 1.2+).`,
					CodeExamples: `# Create an IAM User
aws iam create-user --user-name Alice

# Create a Policy (JSON)
echo '{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": "*",
        "Resource": "*"
    }]
}' > admin-policy.json

# DANGER: The above policy is "AdministratorAccess". Be careful.

# Attach Policy to User
aws iam put-user-policy --user-name Alice --policy-name AdminAccess --policy-document file://admin-policy.json

# Check who called the API (Debug permissions)
aws cloudtrail lookup-events --lookup-attributes AttributeKey=Username,AttributeValue=Alice`,
				},
				{
					Title: "AWS WAF and Shield",
					Content: `AWS WAF and Shield protect applications from common web exploits and DDoS attacks.

**AWS WAF (Web Application Firewall):**
- Protect web applications
- Filter HTTP/HTTPS requests
- Rules for common attacks (SQL injection, XSS)
- Rate-based rules
- Geographic restrictions
- Integration with CloudFront and ALB

**AWS Shield:**
- **Standard**: Free DDoS protection
- **Advanced**: Enhanced DDoS protection ($3000/month)
- Automatic protection for CloudFront and Route 53
- Real-time attack notifications
- Cost protection for scaling

**WAF Rules:**
- **AWS Managed Rules**: Pre-configured rule sets
- **Custom Rules**: Your own rules
- **Rate-based Rules**: Limit requests per IP
- **IP Sets**: Block/allow specific IPs
- **Regex Patterns**: Match request patterns

**Use Cases:**
- Protect against OWASP Top 10
- Rate limiting
- Geographic restrictions
- Bot protection
- DDoS mitigation

**Best Practices:**
- Enable WAF on public-facing resources
- Use AWS managed rules
- Monitor WAF metrics
- Tune rules based on traffic
- Enable Shield Advanced for critical applications`,
					CodeExamples: `# Create IP set
aws wafv2 create-ip-set \\
    --name blocked-ips \\
    --scope REGIONAL \\
    --ip-address-version IPV4 \\
    --addresses 203.0.113.0/24 198.51.100.0/24

# Create WAF web ACL
aws wafv2 create-web-acl \\
    --name my-web-acl \\
    --scope REGIONAL \\
    --default-action Allow={} \\
    --rules file://waf-rules.json \\
    --visibility-config SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=my-web-acl-metrics

# waf-rules.json
[
    {
        "Name": "AWSManagedRulesCommonRuleSet",
        "Priority": 1,
        "Statement": {
            "ManagedRuleGroupStatement": {
                "VendorName": "AWS",
                "Name": "AWSManagedRulesCommonRuleSet"
            }
        },
        "OverrideAction": {
            "None": {}
        },
        "VisibilityConfig": {
            "SampledRequestsEnabled": true,
            "CloudWatchMetricsEnabled": true,
            "MetricName": "CommonRuleSetMetric"
        }
    },
    {
        "Name": "BlockBadIPs",
        "Priority": 2,
        "Statement": {
            "IPSetReferenceStatement": {
                "ARN": "arn:aws:wafv2:us-east-1:123456789012:regional/ipset/blocked-ips/12345678-1234-1234-1234-123456789012"
            }
        },
        "Action": {
            "Block": {}
        },
        "VisibilityConfig": {
            "SampledRequestsEnabled": true,
            "CloudWatchMetricsEnabled": true,
            "MetricName": "BlockBadIPsMetric"
        }
    },
    {
        "Name": "RateLimitRule",
        "Priority": 3,
        "Statement": {
            "RateBasedStatement": {
                "Limit": 2000,
                "AggregateKeyType": "IP"
            }
        },
        "Action": {
            "Block": {}
        },
        "VisibilityConfig": {
            "SampledRequestsEnabled": true,
            "CloudWatchMetricsEnabled": true,
            "MetricName": "RateLimitMetric"
        }
    }
]

# Associate WAF with ALB
aws wafv2 associate-web-acl \\
    --web-acl-arn arn:aws:wafv2:us-east-1:123456789012:regional/webacl/my-web-acl/12345678-1234-1234-1234-123456789012 \\
    --resource-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/my-alb/1234567890123456

# Using boto3
import boto3

wafv2 = boto3.client('wafv2')

# Create IP set
ip_set_response = wafv2.create_ip_set(
    Name='blocked-ips',
    Scope='REGIONAL',
    IPAddressVersion='IPV4',
    Addresses=['203.0.113.0/24', '198.51.100.0/24']
)

# Create web ACL
wafv2.create_web_acl(
    Name='my-web-acl',
    Scope='REGIONAL',
    DefaultAction={'Allow': {}},
    Rules=[
        {
            'Name': 'AWSManagedRulesCommonRuleSet',
            'Priority': 1,
            'Statement': {
                'ManagedRuleGroupStatement': {
                    'VendorName': 'AWS',
                    'Name': 'AWSManagedRulesCommonRuleSet'
                }
            },
            'OverrideAction': {'None': {}},
            'VisibilityConfig': {
                'SampledRequestsEnabled': True,
                'CloudWatchMetricsEnabled': True,
                'MetricName': 'CommonRuleSetMetric'
            }
        }
    ],
    VisibilityConfig={
        'SampledRequestsEnabled': True,
        'CloudWatchMetricsEnabled': True,
        'MetricName': 'my-web-acl-metrics'
    }
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          322,
			Title:       "Cost Optimization",
			Description: "Learn cost optimization strategies: Reserved Instances, Savings Plans, cost analysis, and best practices.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "AWS Cost Optimization Strategies",
					Content: `Cost optimization is an ongoing process to reduce AWS spending while maintaining performance and reliability.

**Cost Optimization Pillars:**
- **Right Sizing**: Match instance size to workload
- **Reserved Instances**: Commit for discounts
- **Savings Plans**: Flexible pricing model
- **Spot Instances**: Use spare capacity
- **Remove Unused Resources**: Clean up idle resources
- **Optimize Storage**: Use appropriate storage classes

**Reserved Instances:**
- 1-year or 3-year terms
- Up to 75% savings vs On-Demand
- Standard, Convertible, Scheduled
- Can be sold in Reserved Instance Marketplace

**Savings Plans:**
- 1-year or 3-year commitments
- Flexible (EC2, Lambda, Fargate)
- Compute Savings Plans (EC2, Lambda, Fargate)
- Up to 72% savings

**Spot Instances:**
- Up to 90% savings
- Interruptible
- Best for fault-tolerant workloads
- Use Spot Fleet for availability

**Cost Monitoring:**
- **Cost Explorer**: Visualize costs
- **Budgets**: Set spending limits
- **Cost Allocation Tags**: Track costs by tag
- **AWS Cost Anomaly Detection**: Detect unusual spending

**Best Practices:**
- Use Cost Explorer regularly
- Set up budgets and alerts
- Tag resources for cost tracking
- Review and optimize regularly
- Use Reserved Instances for predictable workloads
- Right-size instances
- Clean up unused resources`,
					CodeExamples: `# View cost and usage
aws ce get-cost-and-usage \\
    --time-period Start=2024-01-01,End=2024-01-31 \\
    --granularity MONTHLY \\
    --metrics BlendedCost

# Get cost by service
aws ce get-cost-and-usage \\
    --time-period Start=2024-01-01,End=2024-01-31 \\
    --granularity MONTHLY \\
    --metrics BlendedCost \\
    --group-by Type=DIMENSION,Key=SERVICE

# Get cost by tag
aws ce get-cost-and-usage \\
    --time-period Start=2024-01-01,End=2024-01-31 \\
    --granularity MONTHLY \\
    --metrics BlendedCost \\
    --group-by Type=TAG,Key=Environment

# Create budget
aws budgets create-budget \\
    --account-id 123456789012 \\
    --budget file://budget.json

# budget.json
{
    "BudgetName": "MonthlyCostBudget",
    "BudgetLimit": {
        "Amount": "1000",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
        "TagKeyValue": [
            "Environment$Production"
        ]
    }
}

# Purchase Reserved Instance
aws ec2 purchase-reserved-instances-offering \\
    --reserved-instances-offering-id 12345678-1234-1234-1234-123456789012 \\
    --instance-count 1

# Purchase Savings Plan
aws savingsplans purchase-savings-plan \\
    --savings-plan-arn arn:aws:savingsplans::123456789012:savings-plan/12345678-1234-1234-1234-123456789012 \\
    --commitment 1000 \\
    --upfront-payment-amount 0 \\
    --purchase-time 1640995200

# Using boto3
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')
budgets = boto3.client('budgets')

# Get cost and usage
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': start_date.strftime('%Y-%m-%d'),
        'End': end_date.strftime('%Y-%m-%d')
    },
    Granularity='MONTHLY',
    Metrics=['BlendedCost'],
    GroupBy=[
        {'Type': 'DIMENSION', 'Key': 'SERVICE'}
    ]
)

# Create budget
budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'MonthlyCostBudget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'admin@example.com'
                }
            ]
        }
    ]
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          323,
			Title:       "AWS Organizations",
			Description: "Learn Organizations: multi-account management, organizational units, service control policies, and account governance.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "Organizations Fundamentals",
					Content: `AWS Organizations helps you centrally manage and govern multiple AWS accounts.

**Organizations Features:**
- Consolidated billing
- Account management
- Service Control Policies (SCPs)
- Organizational Units (OUs)
- Tag policies
- Cost allocation

**Account Types:**
- **Management Account**: Root account
- **Member Accounts**: Child accounts

**Organizational Units:**
- Group accounts
- Hierarchical structure
- Apply policies to OUs
- Inherit policies

**Service Control Policies:**
- Centralized permissions management
- Applied to accounts or OUs
- Allow or deny services
- Inheritance model

**Best Practices:**
- Use separate accounts for environments
- Implement SCPs for governance
- Use OUs for organization
- Enable all features
- Monitor costs by account`,
					CodeExamples: `# Create organization
aws organizations create-organization --feature-set ALL

# Create organizational unit
aws organizations create-organizational-unit --parent-id r-1234 --name Production

# Create service control policy
aws organizations create-policy --content file://scp.json --name DenyRegions --type SERVICE_CONTROL_POLICY

# Attach policy to OU
aws organizations attach-policy --policy-id p-1234567890 --target-id ou-1234-567890

# Using boto3
import boto3
orgs = boto3.client('organizations')
org = orgs.create_organization(FeatureSet='ALL')
ou = orgs.create_organizational_unit(ParentId='r-1234', Name='Production')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          324,
			Title:       "AWS CloudTrail and Config",
			Description: "Learn CloudTrail and Config: audit logging, compliance monitoring, configuration tracking, and governance.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "CloudTrail and Config Fundamentals",
					Content: `CloudTrail logs API calls, and Config tracks resource configuration changes.

**CloudTrail:**
- API call logging
- Audit trail
- Compliance
- Security analysis
- Event history

**Config:**
- Configuration history
- Compliance checking
- Change notifications
- Resource inventory
- Configuration snapshots

**Use Cases:**
- Security auditing
- Compliance
- Troubleshooting
- Change tracking
- Governance

**Best Practices:**
- Enable CloudTrail in all regions
- Store logs in S3
- Enable log file validation
- Use Config rules
- Monitor compliance`,
					CodeExamples: `# Create CloudTrail
aws cloudtrail create-trail --name my-trail --s3-bucket-name my-trail-bucket

# Start logging
aws cloudtrail start-logging --name my-trail

# Create Config delivery channel
aws configservice put-delivery-channel --delivery-channel file://channel.json

# Start configuration recorder
aws configservice start-configuration-recorder --configuration-recorder-name default

# Using boto3
import boto3
cloudtrail = boto3.client('cloudtrail')
config = boto3.client('config')
trail = cloudtrail.create_trail(Name='my-trail', S3BucketName='my-trail-bucket')
cloudtrail.start_logging(Name='my-trail')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          325,
			Title:       "AWS Well-Architected Framework",
			Description: "Learn the Well-Architected Framework: operational excellence, security, reliability, performance, and cost optimization.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Well-Architected Framework",
					Content: `The AWS Well-Architected Framework provides best practices across five pillars.

**Five Pillars:**
- **Operational Excellence**: Run and monitor systems
- **Security**: Protect data and systems
- **Reliability**: Recover from failures
- **Performance Efficiency**: Use resources efficiently
- **Cost Optimization**: Avoid unnecessary costs

**Design Principles:**
- Stop guessing capacity
- Test at production scale
- Automate changes
- Allow for evolution
- Drive architectures with data

**Best Practices:**
- Regular reviews
- Use Well-Architected Tool
- Document decisions
- Measure and improve
- Share knowledge`,
					CodeExamples: `# Use Well-Architected Tool via console
# Review workloads regularly
# Document architecture decisions
# Measure against best practices`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          326,
			Title:       "Disaster Recovery",
			Description: "Learn disaster recovery strategies: backup, replication, multi-region deployment, and business continuity.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Disaster Recovery Strategies",
					Content: `Disaster recovery ensures business continuity in case of failures.

**Recovery Strategies:**
- **Backup and Restore**: RTO: hours, RPO: 24 hours
- **Pilot Light**: Minimal resources in DR region
- **Warm Standby**: Scaled-down version running
- **Multi-Site Active-Active**: Full production in multiple sites

**RTO and RPO:**
- **RTO**: Recovery Time Objective
- **RPO**: Recovery Point Objective

**AWS Services:**
- S3 Cross-Region Replication
- RDS Multi-AZ and Read Replicas
- DynamoDB Global Tables
- Route 53 health checks
- CloudFormation for automation

**Best Practices:**
- Define RTO and RPO
- Test recovery procedures
- Automate recovery
- Document procedures
- Regular testing`,
					CodeExamples: `# Enable S3 cross-region replication
aws s3api put-bucket-replication --bucket my-bucket --replication-configuration file://replication.json

# Create RDS read replica in another region
aws rds create-db-instance-read-replica --db-instance-identifier mydb-replica --source-db-instance-identifier mydb --region us-west-2

# Using boto3
import boto3
s3 = boto3.client('s3')
rds = boto3.client('rds')
s3.put_bucket_replication(Bucket='my-bucket', ReplicationConfiguration={...})
rds.create_db_instance_read_replica(DBInstanceIdentifier='mydb-replica', SourceDBInstanceIdentifier='mydb')`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
