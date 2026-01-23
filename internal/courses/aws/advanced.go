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
					Title: "VPC Peering and Connectivity",
					Content: `VPC peering allows you to connect VPCs together, enabling resources in different VPCs to communicate.

**VPC Peering:**
- Connect two VPCs in same or different regions
- Private IP communication
- No single point of failure
- No bandwidth bottleneck
- Non-transitive (must peer directly)

**VPN Connections:**
- **Site-to-Site VPN**: Connect on-premises network to VPC
- **Client VPN**: Connect individual users to VPC
- Uses IPsec tunnels
- Supports static and dynamic routing

**AWS Direct Connect:**
- Dedicated network connection to AWS
- Bypass internet
- Consistent network performance
- Lower latency
- Higher bandwidth options

**Transit Gateway:**
- Central hub for VPC and VPN connections
- Simplifies network architecture
- Supports transitive routing
- Route tables for segmentation
- Supports peering across regions

**Best Practices:**
- Use Transit Gateway for complex networks
- Plan CIDR blocks to avoid overlap
- Use route tables for network segmentation
- Monitor network traffic
- Document network architecture`,
					CodeExamples: `# Create VPC peering connection
aws ec2 create-vpc-peering-connection \\
    --vpc-id vpc-12345678 \\
    --peer-vpc-id vpc-87654321 \\
    --peer-region us-west-2

# Accept peering connection
aws ec2 accept-vpc-peering-connection \\
    --vpc-peering-connection-id pcx-12345678

# Add route to peer VPC
aws ec2 create-route \\
    --route-table-id rtb-12345678 \\
    --destination-cidr-block 10.1.0.0/16 \\
    --vpc-peering-connection-id pcx-12345678

# Create VPN connection
aws ec2 create-vpn-connection \\
    --type ipsec.1 \\
    --customer-gateway-id cgw-12345678 \\
    --vpn-gateway-id vgw-12345678 \\
    --options TunnelOptions='[{
        "PreSharedKey": "shared-key-123",
        "TunnelInsideCidr": "169.254.1.0/30"
    }]'

# Create Transit Gateway
aws ec2 create-transit-gateway \\
    --description "Central transit gateway" \\
    --options AmazonSideAsn=64512

# Attach VPC to Transit Gateway
aws ec2 create-transit-gateway-vpc-attachment \\
    --transit-gateway-id tgw-12345678 \\
    --vpc-id vpc-12345678 \\
    --subnet-ids subnet-12345678 subnet-87654321

# Create Transit Gateway route table
aws ec2 create-transit-gateway-route-table \\
    --transit-gateway-id tgw-12345678

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create VPC peering
response = ec2.create_vpc_peering_connection(
    VpcId='vpc-12345678',
    PeerVpcId='vpc-87654321',
    PeerRegion='us-west-2'
)
peering_id = response['VpcPeeringConnection']['VpcPeeringConnectionId']

# Accept peering
ec2.accept_vpc_peering_connection(
    VpcPeeringConnectionId=peering_id
)

# Add route
ec2.create_route(
    RouteTableId='rtb-12345678',
    DestinationCidrBlock='10.1.0.0/16',
    VpcPeeringConnectionId=peering_id
)`,
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
					Title: "Encryption and Secrets Management",
					Content: `AWS provides multiple services for encryption and secrets management.

**AWS KMS (Key Management Service):**
- Managed encryption keys
- Customer Master Keys (CMKs)
- Automatic key rotation
- Integration with many AWS services
- Audit via CloudTrail

**Secrets Manager:**
- Store and retrieve secrets
- Automatic rotation
- Integration with RDS, Redshift, DocumentDB
- Fine-grained access control
- Audit secrets access

**Encryption Types:**
- **Encryption at Rest**: Data stored encrypted
- **Encryption in Transit**: Data encrypted during transmission (TLS/SSL)
- **Client-Side Encryption**: Encrypt before sending to AWS

**Use Cases:**
- Database credentials
- API keys
- Certificates
- Encryption keys

**Best Practices:**
- Enable encryption by default
- Use KMS for key management
- Use Secrets Manager for secrets
- Rotate keys and secrets regularly
- Monitor key usage
- Use least privilege access`,
					CodeExamples: `# Create KMS key
aws kms create-key \\
    --description "Encryption key for S3" \\
    --key-usage ENCRYPT_DECRYPT \\
    --key-spec SYMMETRIC_DEFAULT

# Enable automatic key rotation
aws kms enable-key-rotation --key-id 12345678-1234-1234-1234-123456789012

# Create alias for key
aws kms create-alias \\
    --alias-name alias/my-s3-key \\
    --target-key-id 12345678-1234-1234-1234-123456789012

# Encrypt S3 object with KMS
aws s3 cp file.txt s3://my-bucket/file.txt \\
    --sse aws:kms \\
    --sse-kms-key-id alias/my-s3-key

# Create secret in Secrets Manager
aws secretsmanager create-secret \\
    --name db-credentials \\
    --description "Database credentials" \\
    --secret-string '{"username":"admin","password":"SecretPassword123"}'

# Retrieve secret
aws secretsmanager get-secret-value \\
    --secret-id db-credentials

# Rotate secret automatically
aws secretsmanager rotate-secret \\
    --secret-id db-credentials \\
    --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789012:function:rotate-db-credentials

# Using boto3
import boto3
import json

kms = boto3.client('kms')
secrets = boto3.client('secretsmanager')

# Create KMS key
key_response = kms.create_key(
    Description='Encryption key for S3',
    KeyUsage='ENCRYPT_DECRYPT',
    KeySpec='SYMMETRIC_DEFAULT'
)
key_id = key_response['KeyMetadata']['KeyId']

# Create alias
kms.create_alias(
    AliasName='alias/my-s3-key',
    TargetKeyId=key_id
)

# Create secret
secrets.create_secret(
    Name='db-credentials',
    Description='Database credentials',
    SecretString=json.dumps({
        'username': 'admin',
        'password': 'SecretPassword123'
    })
)

# Retrieve secret
response = secrets.get_secret_value(SecretId='db-credentials')
secret = json.loads(response['SecretString'])
print(f"Username: {secret['username']}")`,
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
