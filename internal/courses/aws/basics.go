package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          300,
			Title:       "AWS Fundamentals",
			Description: "Introduction to AWS: cloud computing basics, AWS global infrastructure, and core services.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "What is AWS?",
					Content: `Amazon Web Services (AWS) is the world's most comprehensive cloud platform. It provides on-demand delivery of IT resources over the Internet with pay-as-you-go pricing.

**In simpler terms:**
Instead of buying, owning, and maintaining physical data centers and servers, you can access functionality like electricity from a utility.

**Why AWS?**
*   **Market Leader:** ~32% market share (2024). If you know AWS, you know the cloud standard.
*   **Scale:** From a single instance to thousands in minutes.
*   **Services:** Over 200+ services. Not just "servers" (EC2), but "serverless" (Lambda), AI/ML, Satellites, and Quantum Computing.

**Global Infrastructure:**
AWS is built around **Regions** and **Availability Zones (AZs)**.

` + "```" + `
Region (e.g., us-east-1 N. Virginia)
│
├── Availability Zone A (us-east-1a) ───┐
│   [ Data Center ]                     │  <-- Low Latency Link
├── Availability Zone B (us-east-1b) ───┤      (< 2ms)
│   [ Data Center ]                     │
└── Availability Zone C (us-east-1c) ───┘
    [ Data Center ]
` + "```" + `

*   **Region:** A physical location around the world (e.g., N. Virginia, Tokyo, Frankfurt).
*   **Availability Zone (AZ):** One or more discrete data centers with redundant power and networking.
*   **Edge Locations:** Points of Presence (PoPs) for caching content (CloudFront) closer to users.

**Key takeaway:** Always design for failure. Deploy your application across *multiple* AZs. If one AZ goes down (flood, fire, power cut), your app keeps running in the others.`,
					CodeExamples: `# Install AWS CLI (The absolute first step)
# https://aws.amazon.com/cli/

# Configure your keys (Access Key ID & Secret Access Key)
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json

# Test it out
$ aws s3 ls
# Lists all your S3 buckets`,
				},
				{
					Title: "AWS Global Infrastructure",
					Content: `AWS has the most extensive global footprint of any cloud provider. It is built on three main layers:

**1. Regions (The "Where")**
A physical location in the world where AWS clusters data centers.
*   **Examples:** 'us-east-1' (N. Virginia), 'eu-central-1' (Frankfurt).
*   **Rule:** Data *never* leaves a region unless you explicitly move it. This is critical for GDPR/compliance.

**2. Availability Zones (The "Resilience")**
Inside every Region, there are 3+ Availability Zones (AZs).
*   **What acts is it:** A distinct location with its own power, cooling, and networking.
*   **Why:** If one AZ fails (fire, flood), the others stay up.
*   **Latency:** < 2ms latency between AZs in the same region.

**Visual Hierarchy:**
` + "```" + `
[ User in London ]
      |
      v
[ Edge Location (London) ]  <-- Caches content for speed
      |
      v
[ Region (eu-west-2 London) ]
      |
      +--- [ AZ A ] <-- App Server
      |
      +--- [ AZ B ] <-- App Server (Backup)
` + "```" + `

**Best Practices:**
1.  **Select the right Region:** Balance **Latency**, **Cost**, and **Features**.
2.  **Multi-AZ is mandatory:** For production, always run in at least 2 AZs.`,
					CodeExamples: `# Find the fastest region for you
# (Ping test from your terminal)
$ ping dynamodb.us-east-1.amazonaws.com
$ ping dynamodb.eu-west-1.amazonaws.com

# List all enabled regions
$ aws ec2 describe-regions --output table`,
				},
				{
					Title: "Core AWS Services",
					Content: `AWS has over 200 services, but you only need to master a few to get started. These are the building blocks of the cloud.

**1. Compute (The "Brains")**
*   **EC2 (Elastic Compute Cloud):** Virtual servers (VMs). You manage the OS.
*   **Lambda:** Serverless code. You upload code, AWS runs it. No servers to manage.

**2. Storage (The "Hard Drive")**
*   **S3 (Simple Storage Service):** Unlimited object storage for files (images, backups). Durable (11 9s).
*   **EBS (Elastic Block Store):** Hard drives for your EC2 instances.

**3. Networking (The "Cables")**
*   **VPC (Virtual Private Cloud):** Your private network in the cloud.
*   **LB (Load Balancer):** Distributes traffic across servers.

**4. Databases (The "Memory")**
*   **RDS (Relational Database Service):** Managed SQL (MySQL, Postgres). AWS handles backups/patches.
*   **DynamoDB:** Fast, flexible NoSQL database. Single-digit millisecond latency.

**5. Security (The "Bouncer")**
*   **IAM (Identity & Access Management):** Controls who can do what (Users, Roles, Permissions).`,
					CodeExamples: `# S3: List buckets
$ aws s3 ls

# EC2: List running instances
$ aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"

# IAM: Who am I?
$ aws sts get-caller-identity`,
				},
				{
					Title: "AWS Account and Billing",
					Content: `Understanding AWS account structure and billing is crucial for managing costs and resources.

**AWS Account:**
- Root account: Primary account owner
- IAM users: Individual users with permissions
- IAM roles: Temporary credentials for services
- Organizations: Manage multiple accounts

**AWS Billing:**
- Pay-as-you-go pricing model
- No upfront costs or long-term commitments
- Billed monthly for usage
- Free tier available for new accounts

**Cost Management:**
- **Cost Explorer**: Visualize and analyze costs
- **Budgets**: Set cost and usage budgets
- **Billing Alerts**: Get notified of spending
- **Reserved Instances**: Save up to 75% with commitments
- **Savings Plans**: Flexible pricing for compute usage

**Free Tier:**
- 12 months free for new accounts
- Always free tier for some services
- Limited usage amounts
- Great for learning and testing

**Best Practices:**
- Set up billing alerts
- Use cost allocation tags
- Review Cost Explorer regularly
- Use Reserved Instances for predictable workloads
- Clean up unused resources`,
					CodeExamples: `# Set up billing alert
# Using AWS Console or CLI
aws budgets create-budget \\
    --account-id 123456789012 \\
    --budget file://budget.json

# budget.json
{
    "BudgetName": "MonthlyBudget",
    "BudgetLimit": {
        "Amount": "100",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
}

# Tag resources for cost tracking
aws ec2 create-tags \\
    --resources i-1234567890abcdef0 \\
    --tags Key=Environment,Value=Production Key=Project,Value=WebApp

# Query costs by tag
aws ce get-cost-and-usage \\
    --time-period Start=2024-01-01,End=2024-01-31 \\
    --granularity MONTHLY \\
    --metrics BlendedCost \\
    --group-by Type=DIMENSION,Key=TAG`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          301,
			Title:       "Identity and Access Management (IAM)",
			Description: "Learn IAM fundamentals: users, groups, roles, policies, and security best practices.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "IAM Fundamentals",
					Content: `Identity and Access Management (IAM) is the foundation of AWS security, providing fine-grained access control to AWS services and resources. IAM enables you to manage who can do what in your AWS account, following the principle of least privilege and ensuring secure access to cloud resources.

**Why IAM Matters:**

**1. Security Foundation:**
- First line of defense for AWS security
- Controls access to all AWS services
- Prevents unauthorized access
- Enables audit trails and compliance

**2. Principle of Least Privilege:**
- Grant minimum permissions needed
- Reduce attack surface
- Limit damage from compromised credentials
- Follow security best practices

**3. Compliance and Auditing:**
- Track who did what (CloudTrail integration)
- Meet compliance requirements
- Audit access patterns
- Demonstrate security controls

**4. Operational Efficiency:**
- Centralized access management
- Easy permission updates
- Role-based access control
- Automated credential management

**IAM Core Concepts:**

**1. Users:**

**What Are IAM Users:**
IAM users represent people or applications that need access to AWS resources. Each user has a unique name and credentials (access keys or password) for authentication.

**User Characteristics:**
- **Unique Identity**: Each user has unique name and credentials
- **Long-term Credentials**: Access keys don't expire (unless rotated)
- **Direct Access**: Users can sign in to AWS Console or use CLI/SDK
- **Individual Permissions**: Each user can have specific permissions
- **MFA Support**: Can enable Multi-Factor Authentication

**When to Use Users:**
- Individual people who need AWS access
- Applications that need long-term credentials (not recommended)
- Service accounts for specific purposes
- Development and testing environments

**User Best Practices:**
- Create separate users for each person/application
- Never share user credentials
- Enable MFA for all users
- Use descriptive user names
- Regularly review and remove unused users
- Rotate access keys regularly

**2. Groups:**

**What Are IAM Groups:**
IAM groups are collections of IAM users. Groups allow you to manage permissions for multiple users at once by attaching policies to the group rather than individual users.

**Group Characteristics:**
- **Collection of Users**: Multiple users can belong to a group
- **Permission Management**: Attach policies to groups
- **Inheritance**: Users inherit group permissions
- **No Nesting**: Groups cannot contain other groups
- **No Direct Access**: Groups cannot sign in or have credentials

**When to Use Groups:**
- Organize users by job function (Developers, Admins, etc.)
- Manage permissions for teams
- Simplify permission management
- Apply consistent permissions

**Group Best Practices:**
- Create groups based on job functions
- Use descriptive group names
- Attach policies to groups, not individual users
- Users can belong to multiple groups
- Regularly review group membership

**Common Group Patterns:**
- **Administrators**: Full access to AWS account
- **Developers**: Access to development resources
- **ReadOnlyUsers**: Read-only access for monitoring
- **BillingUsers**: Access to billing and cost management
- **SecurityAuditors**: Access to security logs and compliance

**3. Roles:**

**What Are IAM Roles:**
IAM roles are similar to users but are intended to be assumed by trusted entities (users, applications, or AWS services). Roles provide temporary credentials that expire automatically, making them more secure than long-term access keys.

**Role Characteristics:**
- **Temporary Credentials**: Credentials expire after session duration
- **Assumable**: Entities assume roles to get credentials
- **Trust Policy**: Defines who can assume the role
- **Permission Policy**: Defines what the role can do
- **No Long-term Credentials**: More secure than users

**When to Use Roles:**
- **EC2 Instances**: Applications running on EC2
- **Lambda Functions**: Serverless functions
- **Cross-Account Access**: Access resources in other accounts
- **Federated Access**: Users from external identity providers
- **AWS Services**: Services that need to access other services

**Role Types:**
- **Service Roles**: Used by AWS services (EC2, Lambda, etc.)
- **Service-Linked Roles**: Predefined roles for specific services
- **Cross-Account Roles**: Access resources in other accounts
- **Federated Roles**: Assumed by external identity providers

**Role Best Practices:**
- Use roles instead of access keys when possible
- Set appropriate session duration
- Use least privilege permissions
- Enable CloudTrail for role assumption tracking
- Use external ID for cross-account access

**4. Policies:**

**What Are IAM Policies:**
Policies are JSON documents that define permissions. They specify what actions are allowed or denied on which resources under what conditions.

**Policy Structure:**

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::my-bucket/*",
                "Condition": {
                    "IpAddress": {
                        "aws:SourceIp": "203.0.113.0/24"
                    }
                }
            }
        ]
    }

**Policy Elements:**
- **Version**: Policy language version (always use "2012-10-17")
- **Statement**: Array of permission statements
- **Effect**: Allow or Deny
- **Action**: API actions (e.g., s3:GetObject, ec2:RunInstances)
- **Resource**: AWS resources (ARNs)
- **Condition**: Optional conditions (IP, time, tags, etc.)

**Policy Types:**

**1. Identity-Based Policies:**
- Attached to users, groups, or roles
- Define what the identity can do
- Most common policy type
- Can be managed or inline

**2. Resource-Based Policies:**
- Attached to resources (S3 buckets, Lambda functions, etc.)
- Define who can access the resource
- Cross-account access possible
- Examples: S3 bucket policies, Lambda resource policies

**3. Permission Boundaries:**
- Set maximum permissions for users/roles
- Cannot grant more than boundary allows
- Useful for delegating permission management
- Applied to users and roles

**4. Service Control Policies (SCPs):**
- Organization-level policies
- Applied to accounts or OUs
- Cannot be overridden by IAM policies
- Used for centralized governance

**5. Session Policies:**
- Temporary policies for role sessions
- Further restrict permissions
- Applied when assuming roles
- Cannot grant more than role's permissions

**Policy Evaluation Logic:**

**How AWS Evaluates Permissions:**
1. **Explicit Deny**: Always wins (highest priority)
2. **Explicit Allow**: Grants permission
3. **Implicit Deny**: Default (no permission)

**Evaluation Order:**
1. Check for explicit Deny
2. Check for explicit Allow
3. Default to Deny if no match

**Important Rules:**
- Deny always overrides Allow
- All policies are evaluated
- Conditions must be met for Allow
- Resource-based policies evaluated separately

**5. Root Account:**

**What Is Root Account:**
The root account is the email address used to create the AWS account. It has complete, unrestricted access to all AWS services and resources.

**Root Account Characteristics:**
- **Complete Access**: Can do anything in the account
- **Cannot Be Restricted**: No IAM policies apply
- **Billing Access**: Always has billing access
- **Account Management**: Can close account, change support plan
- **Should Not Be Used**: For day-to-day operations

**Root Account Best Practices:**
- **Never use for daily operations**: Create IAM users/roles instead
- **Enable MFA**: Critical security requirement
- **Lock away credentials**: Store securely, use only when necessary
- **Monitor usage**: Enable CloudTrail, review root account activity
- **Use only for**: Account management tasks that require root

**IAM Best Practices:**

**1. Never Share Credentials:**
- Each person/application should have own credentials
- Never share access keys or passwords
- Use roles for shared access
- Rotate compromised credentials immediately

**2. Enable MFA Everywhere:**
- Enable MFA on root account (mandatory)
- Enable MFA for privileged users
- Use hardware MFA for high-security accounts
- Virtual MFA apps for convenience

**3. Follow Principle of Least Privilege:**
- Grant minimum permissions needed
- Start with no permissions, add as needed
- Regularly review and remove unused permissions
- Use permission boundaries for delegation

**4. Use Roles Instead of Access Keys:**
- Roles provide temporary credentials
- Access keys are long-term and risky
- Use roles for EC2, Lambda, ECS
- Use roles for cross-account access

**5. Regular Audits:**
- Review IAM users, groups, roles regularly
- Remove unused credentials
- Review CloudTrail logs
- Check for overly permissive policies
- Use Access Analyzer for policy review

**6. Use Managed Policies:**
- AWS managed policies for common use cases
- Customer managed policies for reusability
- Avoid inline policies when possible
- Version control managed policies

**7. Tagging and Organization:**
- Use consistent naming conventions
- Tag IAM resources for organization
- Document policy purposes
- Use groups for team management

**Common Pitfalls:**

**1. Using Root Account:**
- **Problem**: Using root account for daily operations
- **Risk**: Complete account compromise if credentials leaked
- **Solution**: Create IAM users, never use root
- **Example**: Developer using root account credentials

**2. Overly Permissive Policies:**
- **Problem**: Granting "*" (all actions) on "*" (all resources)
- **Risk**: Unauthorized access, compliance violations
- **Solution**: Grant specific permissions only
- **Example**: Policy allowing s3:* on * (all buckets)

**3. Hardcoded Credentials:**
- **Problem**: Access keys in code, config files, repositories
- **Risk**: Credential theft, unauthorized access
- **Solution**: Use IAM roles, Secrets Manager, environment variables
- **Example**: Access keys committed to GitHub

**4. Not Enabling MFA:**
- **Problem**: Single-factor authentication only
- **Risk**: Account takeover if password compromised
- **Solution**: Enable MFA for all users, especially root
- **Example**: Root account without MFA

**5. Not Rotating Credentials:**
- **Problem**: Using same credentials indefinitely
- **Risk**: Compromised credentials remain valid
- **Solution**: Rotate access keys regularly (every 90 days)
- **Example**: Access keys unchanged for years

**6. Ignoring CloudTrail:**
- **Problem**: No visibility into IAM activity
- **Risk**: Undetected unauthorized access
- **Solution**: Enable CloudTrail, monitor logs
- **Example**: No audit trail for IAM changes

**Real-World Examples:**

**Startup Scenario:**
- Create IAM users for each developer
- Group developers into "Developers" group
- Attach managed policies (PowerUserAccess)
- Enable MFA for all users
- Use roles for applications

**Enterprise Scenario:**
- Multiple accounts via Organizations
- Service Control Policies for governance
- Cross-account roles for access
- Federated access via SAML/OIDC
- Centralized IAM management

**Microservices Architecture:**
- Each service has its own IAM role
- Roles have minimal permissions
- Use VPC endpoints to reduce permissions
- Monitor role usage with CloudTrail
- Automate role creation with CloudFormation

**Conclusion:**

IAM is the cornerstone of AWS security. By understanding users, groups, roles, and policies, and following security best practices, you can build a secure, compliant, and manageable AWS environment. Remember:

- **Never use root account** for daily operations
- **Enable MFA** everywhere, especially root
- **Follow least privilege** principle
- **Use roles** instead of access keys
- **Regular audits** and credential rotation
- **Monitor** IAM activity with CloudTrail

Proper IAM configuration is not optional—it's essential for secure cloud operations. Invest time in designing a good IAM structure from the start, and you'll avoid security issues and compliance problems down the road.`,
					CodeExamples: `# Create IAM user
aws iam create-user --user-name john-doe

# Create IAM group
aws iam create-group --group-name Developers

# Add user to group
aws iam add-user-to-group \\
    --user-name john-doe \\
    --group-name Developers

# Create IAM policy
aws iam create-policy \\
    --policy-name S3ReadOnly \\
    --policy-document file://policy.json

# policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket/*",
                "arn:aws:s3:::my-bucket"
            ]
        }
    ]
}

# Attach policy to group
aws iam attach-group-policy \\
    --group-name Developers \\
    --policy-arn arn:aws:iam::123456789012:policy/S3ReadOnly

# Create IAM role
aws iam create-role \\
    --role-name EC2S3Access \\
    --assume-role-policy-document file://trust-policy.json

# trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}`,
				},
				{
					Title: "IAM Policies and Permissions",
					Content: `IAM policies define what actions can be performed on which resources under what conditions.

**Policy Elements:**
- **Effect**: Allow or Deny
- **Action**: API operations (e.g., s3:GetObject)
- **Resource**: ARN of the resource
- **Condition**: Optional conditions (time, IP, etc.)

**Policy Evaluation:**
1. Explicit Deny always wins
2. Explicit Allow grants permission
3. Implicit Deny (default) denies access

**Common Policy Patterns:**
- **Read-only access**: List and Get operations
- **Write access**: Put, Delete, Create operations
- **Admin access**: Full access with *
- **Conditional access**: Based on conditions

**Managed Policies:**
- AWS managed: Predefined by AWS
- Customer managed: Created by you
- Inline policies: Embedded in user/role/group

**Best Practices:**
- Use managed policies when possible
- Grant minimum required permissions
- Use conditions for additional security
- Test policies before applying
- Document custom policies`,
					CodeExamples: `# Example: S3 bucket read-only policy
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ]
        }
    ]
}

# Example: EC2 full access with condition
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "ec2:*",
            "Resource": "*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}

# Example: Time-based access
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThan": {
                    "aws:CurrentTime": "09:00Z"
                },
                "DateLessThan": {
                    "aws:CurrentTime": "17:00Z"
                }
            }
        }
    ]
}

# Example: Tag-based access
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "ec2:*",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "ec2:ResourceTag/Owner": "${aws:username}"
                }
            }
        }
    ]
}`,
				},
				{
					Title: "IAM Roles and Cross-Account Access",
					Content: `IAM roles provide temporary credentials and enable secure access across AWS accounts and services.

**IAM Roles:**
- Temporary credentials (no long-term keys)
- Assumed by users, services, or applications
- Defined by trust policy (who can assume)
- Defined by permission policy (what they can do)

**Use Cases:**
- EC2 instances accessing S3
- Lambda functions accessing DynamoDB
- Cross-account access
- Federated access (SAML, OIDC)

**Role Types:**
- **Service roles**: Used by AWS services
- **Service-linked roles**: Predefined for specific services
- **Cross-account roles**: Access across accounts

**Assume Role Process:**
1. Entity requests to assume role
2. IAM checks trust policy
3. If allowed, temporary credentials issued
4. Credentials expire after session duration

**Best Practices:**
- Use roles instead of access keys when possible
- Set appropriate session duration
- Use external ID for cross-account access
- Enable CloudTrail for role assumption tracking`,
					CodeExamples: `# Create role for EC2 to access S3
aws iam create-role \\
    --role-name EC2S3Role \\
    --assume-role-policy-document file://ec2-trust.json

# ec2-trust.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

# Attach S3 access policy
aws iam attach-role-policy \\
    --role-name EC2S3Role \\
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile \\
    --instance-profile-name EC2S3Profile

# Add role to instance profile
aws iam add-role-to-instance-profile \\
    --instance-profile-name EC2S3Profile \\
    --role-name EC2S3Role

# Cross-account role
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT-ID:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "sts:ExternalId": "unique-external-id"
                }
            }
        }
    ]
}

# Assume role from application
import boto3

sts = boto3.client('sts')
response = sts.assume_role(
    RoleArn='arn:aws:iam::123456789012:role/CrossAccountRole',
    RoleSessionName='session1',
    ExternalId='unique-external-id'
)

credentials = response['Credentials']
s3 = boto3.client(
    's3',
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          302,
			Title:       "Amazon EC2",
			Description: "Master EC2: instances, AMIs, security groups, key pairs, and instance management.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "EC2 Fundamentals",
					Content: `Amazon Elastic Compute Cloud (EC2) is AWS's core compute service, providing resizable virtual servers in the cloud. EC2 eliminates the need to invest in hardware upfront, allowing you to launch virtual servers, configure security and networking, and manage storage with complete control.

**Why EC2 Matters:**

**1. Flexibility:**
- Choose from hundreds of instance types
- Select operating systems and software
- Scale capacity up or down instantly
- Pay only for what you use

**2. Control:**
- Root access to instances
- Complete control over software stack
- Custom configurations
- Full administrative access

**3. Integration:**
- Integrates with all AWS services
- Works with VPC, S3, RDS, and more
- Part of comprehensive cloud architecture
- Enables hybrid cloud deployments

**4. Cost-Effective:**
- No upfront costs
- Multiple pricing options
- Pay only for running instances
- Significant cost savings vs on-premises

**EC2 Core Concepts:**

**1. Instances:**

**What Are EC2 Instances:**
An EC2 instance is a virtual server running in AWS's cloud. It's similar to a physical server but virtualized, allowing you to launch, configure, and manage servers programmatically.

**Instance Characteristics:**
- **Virtual Servers**: Running on physical AWS hardware
- **Configurable**: Choose CPU, memory, storage, networking
- **Scalable**: Resize or add instances as needed
- **Temporary**: Can be stopped, terminated, or replaced
- **Isolated**: Each instance is isolated from others

**Instance Lifecycle:**

    Launch → Pending → Running → Stopping → Stopped → Terminating → Terminated
                                  ↓
                             Rebooting → Running

**2. AMIs (Amazon Machine Images):**

**What Are AMIs:**
An AMI is a template that contains the software configuration (operating system, application server, applications) required to launch an instance.

**AMI Components:**
- **Root Volume**: Contains OS and applications
- **Launch Permissions**: Who can use the AMI
- **Block Device Mapping**: Volume attachments
- **Kernel and RAM Disk**: For paravirtualized instances

**AMI Types:**
- **Public AMIs**: Provided by AWS or community
- **Private AMIs**: Your custom AMIs
- **Marketplace AMIs**: Pre-configured software (may have fees)
- **Shared AMIs**: Shared by other AWS accounts

**AMI Best Practices:**
- Use latest AMIs for security patches
- Create custom AMIs for consistent deployments
- Test AMIs before production use
- Document AMI contents and purpose
- Keep AMI library organized

**3. Instance Types:**

**Understanding Instance Families:**

**General Purpose (t3, t4g, m5, m6i):**
- **Use Cases**: Web servers, small databases, development
- **Balance**: Compute, memory, networking
- **Examples**: t3.micro, m5.large, m6i.xlarge
- **Characteristics**: Burstable or consistent performance

**Compute Optimized (c5, c6i, c7i):**
- **Use Cases**: High-performance computing, batch processing, gaming servers
- **Focus**: High CPU performance
- **Examples**: c5.xlarge, c6i.2xlarge
- **Characteristics**: Latest generation processors

**Memory Optimized (r5, r6i, x1, x2ie):**
- **Use Cases**: Databases, in-memory caches, big data
- **Focus**: Large memory capacity
- **Examples**: r5.2xlarge, x1.32xlarge
- **Characteristics**: High memory-to-vCPU ratio

**Storage Optimized (i3, i4i, d2, d3):**
- **Use Cases**: NoSQL databases, data warehousing, file systems
- **Focus**: High I/O performance, large storage
- **Examples**: i3.xlarge, d3.2xlarge
- **Characteristics**: NVMe SSD, high IOPS

**Accelerated Computing (p3, p4, g4, g5):**
- **Use Cases**: Machine learning, graphics rendering, video encoding
- **Focus**: GPUs, FPGAs, specialized hardware
- **Examples**: p3.2xlarge, g4dn.xlarge
- **Characteristics**: GPU instances, specialized workloads

**Instance Type Naming:**
- **Format**: family.generation.size
- **Example**: m5.2xlarge
  - m = General purpose
  - 5 = 5th generation
  - 2xlarge = Size (8 vCPUs, 32 GB RAM)

**Choosing Instance Types:**
- **Analyze Workload**: CPU, memory, storage, network needs
- **Start Small**: Begin with smaller instances, scale up
- **Right-Size**: Match instance to actual needs
- **Consider Cost**: Balance performance and cost
- **Test Performance**: Benchmark different types

**4. Security Groups:**

**What Are Security Groups:**
Security groups act as virtual firewalls controlling inbound and outbound traffic for EC2 instances. They are stateful, meaning return traffic is automatically allowed.

**Security Group Rules:**
- **Type**: Protocol (TCP, UDP, ICMP, etc.)
- **Port Range**: Single port or range
- **Source/Destination**: IP address, CIDR block, or security group
- **Description**: Optional rule description

**Security Group Characteristics:**
- **Stateful**: Return traffic automatically allowed
- **Default Deny**: All traffic denied by default
- **Multiple Groups**: Instance can belong to multiple groups
- **VPC-Specific**: Security groups belong to VPCs

**Security Group Best Practices:**
- Follow principle of least privilege
- Use specific IP ranges, not 0.0.0.0/0
- Separate security groups by function
- Document security group rules
- Review and update regularly

**5. Key Pairs:**

**What Are Key Pairs:**
Key pairs are used to securely connect to EC2 instances. They consist of a public key (stored by AWS) and a private key (you keep secure).

**Key Pair Types:**
- **RSA**: Traditional, widely supported
- **ED25519**: Modern, more secure (recommended)
- **EC2 Key Pair**: Managed by AWS
- **Import Key Pair**: Import your own public key

**Key Pair Best Practices:**
- Store private keys securely (never share)
- Use different key pairs for different environments
- Rotate key pairs regularly
- Use SSH agent for convenience
- Consider using Systems Manager Session Manager (no keys needed)

**6. EBS Volumes:**

**What Are EBS Volumes:**
Elastic Block Store (EBS) provides persistent block storage volumes for EC2 instances. Unlike instance store (ephemeral), EBS volumes persist independently of instance lifecycle.

**EBS Volume Types:**
- **gp3**: Latest general purpose SSD (default, recommended)
- **gp2**: Previous generation general purpose SSD
- **io1/io2**: Provisioned IOPS SSD (high performance)
- **st1**: Throughput Optimized HDD
- **sc1**: Cold HDD (lowest cost)

**EBS Characteristics:**
- **Persistent**: Survives instance termination
- **Attachable**: Can attach/detach volumes
- **Resizable**: Can increase size dynamically
- **Snapshottable**: Create backups
- **Encryptable**: Encryption at rest

**Pricing Models:**

**1. On-Demand Instances:**
- **Pricing**: Pay per hour/second
- **Commitment**: None
- **Use Cases**: Short-term, unpredictable workloads
- **Benefits**: No upfront costs, flexible
- **Drawbacks**: Highest hourly rate

**2. Reserved Instances:**
- **Pricing**: Up to 75% discount vs On-Demand
- **Commitment**: 1 or 3 years
- **Types**: Standard, Convertible, Scheduled
- **Use Cases**: Predictable, steady-state workloads
- **Benefits**: Significant cost savings
- **Drawbacks**: Upfront payment, less flexibility

**3. Spot Instances:**
- **Pricing**: Up to 90% discount vs On-Demand
- **Commitment**: None (can be interrupted)
- **Use Cases**: Fault-tolerant, flexible workloads
- **Benefits**: Lowest cost
- **Drawbacks**: Can be interrupted with 2-minute notice

**4. Savings Plans:**
- **Pricing**: Up to 72% discount
- **Commitment**: 1 or 3 years
- **Types**: Compute, EC2 Instance
- **Use Cases**: Flexible compute usage
- **Benefits**: Flexible, automatic discount
- **Drawbacks**: Less discount than RIs

**Best Practices:**

**1. Instance Management:**
- **Right-Size**: Match instance to workload needs
- **Use Auto Scaling**: Automatically adjust capacity
- **Tag Instances**: Organize and track resources
- **Monitor Performance**: Use CloudWatch metrics
- **Terminate Unused**: Stop or terminate idle instances

**2. Security:**
- **Use Security Groups**: Control network access
- **Enable Termination Protection**: Prevent accidental termination
- **Use IAM Roles**: Instead of access keys
- **Enable VPC Flow Logs**: Monitor network traffic
- **Regular Updates**: Keep AMIs and software updated

**3. Cost Optimization:**
- **Use Reserved Instances**: For predictable workloads
- **Use Spot Instances**: For fault-tolerant workloads
- **Right-Size Instances**: Don't over-provision
- **Monitor Costs**: Use Cost Explorer
- **Clean Up**: Terminate unused instances

**4. High Availability:**
- **Multi-AZ Deployment**: Deploy across Availability Zones
- **Use Load Balancers**: Distribute traffic
- **Auto Scaling**: Handle traffic spikes
- **Health Checks**: Monitor instance health
- **Backup Strategy**: Regular snapshots and AMIs

**5. Performance:**
- **Choose Right Instance Type**: Match to workload
- **Use Enhanced Networking**: For better network performance
- **Optimize Storage**: Use appropriate EBS volume types
- **Monitor Metrics**: Track CPU, memory, network, disk
- **Benchmark**: Test performance before production

**Common Pitfalls:**

**1. Over-Provisioning:**
- **Problem**: Choosing instance types too large
- **Impact**: Wasted costs, poor resource utilization
- **Solution**: Start small, monitor, right-size based on metrics
- **Example**: Using m5.4xlarge when t3.medium would suffice

**2. Under-Provisioning:**
- **Problem**: Instance too small for workload
- **Impact**: Poor performance, user experience issues
- **Solution**: Monitor metrics, scale up when needed
- **Example**: Running database on t2.micro

**3. Not Using Auto Scaling:**
- **Problem**: Manual scaling, over-provisioning for peaks
- **Impact**: High costs or poor performance during spikes
- **Solution**: Implement Auto Scaling groups
- **Example**: Running 10 instances 24/7 for peak traffic 2 hours/day

**4. Poor Security Group Configuration:**
- **Problem**: Overly permissive rules (0.0.0.0/0)
- **Impact**: Security vulnerabilities, unauthorized access
- **Solution**: Use least privilege, specific IP ranges
- **Example**: Allowing SSH (port 22) from 0.0.0.0/0

**5. Not Backing Up:**
- **Problem**: No EBS snapshots or AMIs
- **Impact**: Data loss, long recovery times
- **Solution**: Regular snapshots, automated backups
- **Example**: Instance failure with no backups

**6. Ignoring Costs:**
- **Problem**: Leaving instances running 24/7
- **Impact**: Unexpected high bills
- **Solution**: Stop/terminate when not needed, use scheduling
- **Example**: Development instances running over weekend

**Real-World Examples:**

**Web Application:**
- **Architecture**: Auto Scaling group behind Application Load Balancer
- **Instance Type**: t3.medium (general purpose)
- **Deployment**: Multiple AZs for high availability
- **Scaling**: Auto Scaling based on CPU/memory
- **Cost**: On-Demand for variable traffic, Reserved for baseline

**Database Server:**
- **Architecture**: RDS Multi-AZ (managed) or EC2 with replication
- **Instance Type**: r5.2xlarge (memory optimized)
- **Storage**: io2 EBS volumes for high IOPS
- **Backup**: Automated daily snapshots
- **Cost**: Reserved Instances for predictable workload

**Batch Processing:**
- **Architecture**: Spot Fleet for cost optimization
- **Instance Type**: c5.xlarge (compute optimized)
- **Deployment**: Spot Instances with On-Demand fallback
- **Fault Tolerance**: Checkpointing, retry logic
- **Cost**: Up to 90% savings with Spot Instances

**Machine Learning Training:**
- **Architecture**: GPU instances for training
- **Instance Type**: p3.2xlarge (GPU accelerated)
- **Storage**: High-throughput EBS or instance store
- **Scaling**: Multiple instances for distributed training
- **Cost**: Spot Instances for training workloads

**Conclusion:**

EC2 is the foundation of AWS compute services, providing flexible, scalable, and cost-effective virtual servers. By understanding instance types, pricing models, security, and best practices, you can build reliable, secure, and cost-optimized applications.

Key takeaways:
- **Choose right instance type** for your workload
- **Use multiple pricing models** to optimize costs
- **Implement security best practices** (security groups, IAM roles)
- **Design for high availability** (multi-AZ, Auto Scaling)
- **Monitor and optimize** continuously

EC2's flexibility and integration with other AWS services make it the go-to choice for most compute workloads. Whether you're running simple web applications or complex distributed systems, EC2 provides the foundation you need.`,
					CodeExamples: `# Launch EC2 instance
aws ec2 run-instances \\
    --image-id ami-0c55b159cbfafe1f0 \\
    --instance-type t2.micro \\
    --key-name my-key-pair \\
    --security-group-ids sg-12345678 \\
    --subnet-id subnet-12345678 \\
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'

# List running instances
aws ec2 describe-instances \\
    --filters "Name=instance-state-name,Values=running"

# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Terminate instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Create AMI from instance
aws ec2 create-image \\
    --instance-id i-1234567890abcdef0 \\
    --name "my-web-server-ami" \\
    --description "Web server AMI"

# Using boto3 (Python)
import boto3

ec2 = boto3.client('ec2')

# Launch instance
response = ec2.run_instances(
    ImageId='ami-0c55b159cbfafe1f0',
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro',
    KeyName='my-key-pair',
    SecurityGroupIds=['sg-12345678']
)

instance_id = response['Instances'][0]['InstanceId']
print(f"Launched instance: {instance_id}")`,
				},
				{
					Title: "EC2 Security Groups and Networking",
					Content: `Security groups act as virtual firewalls controlling inbound and outbound traffic for EC2 instances.

**Security Groups:**
- Stateful: Return traffic automatically allowed
- Default deny: All traffic denied by default
- Rules: Allow specific traffic
- Multiple groups: Instance can belong to multiple groups

**Security Group Rules:**
- **Type**: Protocol (TCP, UDP, ICMP, etc.)
- **Port Range**: Port or port range
- **Source/Destination**: IP address or security group
- **Description**: Optional rule description

**Default Security Groups:**
- All outbound traffic allowed
- No inbound traffic allowed
- Can be modified

**Network Interfaces:**
- Primary network interface (eth0)
- Additional ENIs can be attached
- Each ENI can have multiple IP addresses
- Can be moved between instances

**Elastic IP Addresses:**
- Static public IP address
- Can be associated with instance
- Released when not in use
- Charges apply if not associated

**Best Practices:**
- Follow principle of least privilege
- Use specific IP ranges, not 0.0.0.0/0
- Separate security groups by function
- Document security group rules
- Review security groups regularly`,
					CodeExamples: `# Create security group
aws ec2 create-security-group \\
    --group-name web-server-sg \\
    --description "Security group for web servers" \\
    --vpc-id vpc-12345678

# Add inbound rule (HTTP)
aws ec2 authorize-security-group-ingress \\
    --group-id sg-12345678 \\
    --protocol tcp \\
    --port 80 \\
    --cidr 0.0.0.0/0

# Add inbound rule (HTTPS)
aws ec2 authorize-security-group-ingress \\
    --group-id sg-12345678 \\
    --protocol tcp \\
    --port 443 \\
    --cidr 0.0.0.0/0

# Add inbound rule (SSH from specific IP)
aws ec2 authorize-security-group-ingress \\
    --group-id sg-12345678 \\
    --protocol tcp \\
    --port 22 \\
    --cidr 203.0.113.0/24

# Add inbound rule from another security group
aws ec2 authorize-security-group-ingress \\
    --group-id sg-12345678 \\
    --protocol tcp \\
    --port 3306 \\
    --source-group sg-87654321

# Remove rule
aws ec2 revoke-security-group-ingress \\
    --group-id sg-12345678 \\
    --protocol tcp \\
    --port 80 \\
    --cidr 0.0.0.0/0

# Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# Associate Elastic IP with instance
aws ec2 associate-address \\
    --instance-id i-1234567890abcdef0 \\
    --allocation-id eipalloc-12345678

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create security group
response = ec2.create_security_group(
    GroupName='web-server-sg',
    Description='Security group for web servers',
    VpcId='vpc-12345678'
)
sg_id = response['GroupId']

# Add rules
ec2.authorize_security_group_ingress(
    GroupId=sg_id,
    IpPermissions=[
        {
            'IpProtocol': 'tcp',
            'FromPort': 80,
            'ToPort': 80,
            'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTP'}]
        },
        {
            'IpProtocol': 'tcp',
            'FromPort': 443,
            'ToPort': 443,
            'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTPS'}]
        }
    ]
)`,
				},
				{
					Title: "EBS Volumes and Snapshots",
					Content: `Amazon Elastic Block Store (EBS) provides persistent block storage volumes for EC2 instances.

**EBS Volume Types:**
- **gp3**: Latest generation general purpose SSD (default)
- **gp2**: General purpose SSD (legacy)
- **io1/io2**: Provisioned IOPS SSD (high performance)
- **st1**: Throughput Optimized HDD
- **sc1**: Cold HDD (lowest cost)

**EBS Features:**
- Persistent storage independent of instance lifecycle
- Can attach/detach volumes
- Create snapshots for backup
- Encrypt volumes with KMS
- Resize volumes dynamically

**Snapshots:**
- Point-in-time backups of volumes
- Incremental (only changed blocks)
- Can create volumes from snapshots
- Can copy snapshots across regions
- Can share snapshots with other accounts

**Volume Operations:**
- Create volume
- Attach to instance
- Detach from instance
- Create snapshot
- Restore from snapshot
- Modify volume (size, type, IOPS)

**Best Practices:**
- Use gp3 for most workloads
- Enable encryption by default
- Create regular snapshots
- Monitor volume performance
- Right-size volumes
- Use appropriate volume type for workload`,
					CodeExamples: `# Create EBS volume
aws ec2 create-volume \\
    --availability-zone us-east-1a \\
    --size 100 \\
    --volume-type gp3 \\
    --encrypted \\
    --kms-key-id alias/my-key

# Attach volume to instance
aws ec2 attach-volume \\
    --volume-id vol-1234567890abcdef0 \\
    --instance-id i-1234567890abcdef0 \\
    --device /dev/sdf

# Create snapshot
aws ec2 create-snapshot \\
    --volume-id vol-1234567890abcdef0 \\
    --description "Daily backup snapshot"

# Create volume from snapshot
aws ec2 create-volume \\
    --availability-zone us-east-1a \\
    --snapshot-id snap-1234567890abcdef0 \\
    --volume-type gp3

# Modify volume (increase size)
aws ec2 modify-volume \\
    --volume-id vol-1234567890abcdef0 \\
    --size 200

# Copy snapshot to another region
aws ec2 copy-snapshot \\
    --source-region us-east-1 \\
    --source-snapshot-id snap-1234567890abcdef0 \\
    --region us-west-2 \\
    --description "Copied snapshot"

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create volume
volume = ec2.create_volume(
    AvailabilityZone='us-east-1a',
    Size=100,
    VolumeType='gp3',
    Encrypted=True,
    KmsKeyId='alias/my-key'
)
volume_id = volume['VolumeId']

# Wait for volume to be available
waiter = ec2.get_waiter('volume_available')
waiter.wait(VolumeIds=[volume_id])

# Attach volume
ec2.attach_volume(
    VolumeId=volume_id,
    InstanceId='i-1234567890abcdef0',
    Device='/dev/sdf'
)

# Create snapshot
snapshot = ec2.create_snapshot(
    VolumeId=volume_id,
    Description='Daily backup snapshot'
)
print(f"Snapshot ID: {snapshot['SnapshotId']}")`,
				},
				{
					Title: "EC2 Instance Metadata and User Data",
					Content: `EC2 instance metadata provides information about the instance, and user data allows scripts to run at launch.

**Instance Metadata:**
- Information about the instance
- Available at http://169.254.169.254/latest/meta-data/
- Includes instance ID, type, AMI, security groups, etc.
- IMDSv2 (Instance Metadata Service Version 2) for security
- Can disable metadata service

**User Data:**
- Scripts that run when instance launches
- Can be shell scripts, cloud-init directives, etc.
- Runs as root user
- Limited to 16KB
- Can retrieve user data from metadata service

**Common Use Cases:**
- Install software packages
- Configure applications
- Download files
- Set up services
- Bootstrap applications

**Metadata Categories:**
- **meta-data/**: Instance information
- **user-data**: Launch scripts
- **dynamic/**: Dynamic data (instance identity document)
- **latest/**: Latest version (recommended)

**Security:**
- Use IMDSv2 (token-based)
- Restrict metadata access with IAM roles
- Don't store secrets in user data
- Use Secrets Manager for sensitive data

**Best Practices:**
- Use IMDSv2 for security
- Keep user data scripts simple
- Use cloud-init for complex setup
- Test user data scripts
- Use IAM roles instead of access keys`,
					CodeExamples: `# Retrieve instance metadata (IMDSv1)
curl http://169.254.169.254/latest/meta-data/instance-id
curl http://169.254.169.254/latest/meta-data/instance-type
curl http://169.254.169.254/latest/meta-data/ami-id
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Retrieve instance metadata (IMDSv2 - secure)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id

# Retrieve user data
curl http://169.254.169.254/latest/user-data

# Launch instance with user data
aws ec2 run-instances \\
    --image-id ami-0c55b159cbfafe1f0 \\
    --instance-type t2.micro \\
    --user-data file://user-data.sh \\
    --key-name my-key-pair

# user-data.sh
#!/bin/bash
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd
echo "<h1>Hello from EC2</h1>" > /var/www/html/index.html

# Enable IMDSv2 only
aws ec2 modify-instance-metadata-options \\
    --instance-id i-1234567890abcdef0 \\
    --http-tokens required \\
    --http-endpoint enabled

# Using boto3
import boto3
import requests

ec2 = boto3.client('ec2')

# Launch instance with user data
with open('user-data.sh', 'r') as f:
    user_data = f.read()

response = ec2.run_instances(
    ImageId='ami-0c55b159cbfafe1f0',
    InstanceType='t2.micro',
    UserData=user_data,
    MinCount=1,
    MaxCount=1
)

# Retrieve metadata from running instance (Python)
# Note: This runs on the EC2 instance, not locally
import requests

def get_metadata(path):
    token_url = 'http://169.254.169.254/latest/api/token'
    token_headers = {'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
    token = requests.put(token_url, headers=token_headers).text
    
    metadata_url = f'http://169.254.169.254/latest/{path}'
    metadata_headers = {'X-aws-ec2-metadata-token': token}
    return requests.get(metadata_url, headers=metadata_headers).text

instance_id = get_metadata('meta-data/instance-id')
instance_type = get_metadata('meta-data/instance-type')
print(f"Instance ID: {instance_id}")
print(f"Instance Type: {instance_type}")`,
				},
				{
					Title: "EC2 Placement Groups and Enhanced Networking",
					Content: `Placement groups control instance placement, and enhanced networking provides better network performance.

**Placement Group Types:**
- **Cluster**: Low latency, high throughput (same rack)
- **Spread**: Maximum availability (separate hardware)
- **Partition**: Large distributed workloads (logical partitions)

**Cluster Placement Groups:**
- Instances in same Availability Zone
- Low network latency
- High network throughput
- Best for HPC, big data
- Single point of failure risk

**Spread Placement Groups:**
- Instances on distinct hardware
- Maximum availability
- Up to 7 instances per AZ
- Best for critical applications
- No single point of failure

**Partition Placement Groups:**
- Logical partitions
- Up to 7 partitions per AZ
- Multiple instances per partition
- Best for distributed systems
- Isolated failure domains

**Enhanced Networking:**
- **SR-IOV**: Single Root I/O Virtualization
- Lower latency
- Higher throughput
- Lower jitter
- Available on certain instance types

**Elastic Network Adapter (ENA):**
- Up to 100 Gbps
- Lower latency
- Available on most instance types
- Enabled by default on supported types

**Best Practices:**
- Use cluster placement for HPC workloads
- Use spread placement for critical apps
- Use partition placement for distributed systems
- Enable enhanced networking when available
- Choose appropriate instance types`,
					CodeExamples: `# Create cluster placement group
aws ec2 create-placement-group \\
    --group-name hpc-cluster \\
    --strategy cluster

# Create spread placement group
aws ec2 create-placement-group \\
    --group-name critical-apps \\
    --strategy spread

# Create partition placement group
aws ec2 create-placement-group \\
    --group-name distributed-system \\
    --strategy partition \\
    --partition-count 3

# Launch instance in placement group
aws ec2 run-instances \\
    --image-id ami-0c55b159cbfafe1f0 \\
    --instance-type c5.9xlarge \\
    --placement GroupName=hpc-cluster \\
    --key-name my-key-pair

# Describe placement group
aws ec2 describe-placement-groups \\
    --group-names hpc-cluster

# Enable enhanced networking (SR-IOV)
# Requires instance with ENA support
# Check if ENA is enabled
aws ec2 describe-instance-types \\
    --instance-types c5.xlarge \\
    --query 'InstanceTypes[0].NetworkInfo.EnaSupport'

# Launch instance with ENA (automatic on supported types)
aws ec2 run-instances \\
    --image-id ami-0c55b159cbfafe1f0 \\
    --instance-type c5.xlarge \\
    --network-interfaces '[{
        "DeviceIndex": 0,
        "SubnetId": "subnet-12345678",
        "Groups": ["sg-12345678"],
        "InterfaceType": "efa"
    }]'

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create cluster placement group
ec2.create_placement_group(
    GroupName='hpc-cluster',
    Strategy='cluster'
)

# Launch instances in placement group
for i in range(4):
    ec2.run_instances(
        ImageId='ami-0c55b159cbfafe1f0',
        InstanceType='c5.9xlarge',
        Placement={'GroupName': 'hpc-cluster'},
        MinCount=1,
        MaxCount=1
    )

# Check ENA support
response = ec2.describe_instance_types(
    InstanceTypes=['c5.xlarge']
)
ena_support = response['InstanceTypes'][0]['NetworkInfo']['EnaSupport']
print(f"ENA Support: {ena_support}")`,
				},
				{
					Title: "EC2 Spot Instances and Spot Fleets",
					Content: `Spot Instances allow you to bid on unused EC2 capacity at up to 90% discount.

**Spot Instances:**
- Use spare EC2 capacity
- Up to 90% savings vs On-Demand
- Can be interrupted with 2-minute notice
- Best for fault-tolerant workloads
- Not suitable for persistent workloads

**Spot Instance Concepts:**
- **Spot Price**: Current market price
- **Max Price**: Maximum you're willing to pay
- **Spot Request**: Request for spot capacity
- **Interruption**: When AWS needs capacity back

**Spot Instance Interruption:**
- 2-minute warning before interruption
- Instance receives termination notice
- Can check interruption notice in metadata
- Automatic stop/terminate based on request type

**Spot Fleets:**
- Manage collection of Spot Instances
- Multiple instance types and AZs
- Automatic capacity management
- Diversification strategies
- Target capacity management

**Spot Fleet Strategies:**
- **lowestPrice**: Lowest cost
- **diversified**: Distribute across pools
- **capacityOptimized**: Best capacity availability

**Use Cases:**
- Batch processing
- Data analysis
- CI/CD workloads
- Web crawling
- Image processing
- Any fault-tolerant workload

**Best Practices:**
- Use for fault-tolerant workloads only
- Implement checkpointing
- Handle interruptions gracefully
- Use Spot Fleets for availability
- Monitor spot prices
- Use multiple instance types`,
					CodeExamples: `# Request spot instance
aws ec2 request-spot-instances \\
    --spot-price "0.05" \\
    --instance-count 1 \\
    --type "one-time" \\
    --launch-specification file://launch-spec.json

# launch-spec.json
{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "t2.micro",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-12345678"],
    "SubnetId": "subnet-12345678"
}

# Create persistent spot request
aws ec2 request-spot-instances \\
    --spot-price "0.05" \\
    --instance-count 1 \\
    --type "persistent" \\
    --launch-specification file://launch-spec.json

# Describe spot price history
aws ec2 describe-spot-price-history \\
    --instance-types t2.micro \\
    --product-descriptions "Linux/UNIX" \\
    --max-items 10

# Cancel spot request
aws ec2 cancel-spot-instance-requests \\
    --spot-instance-request-ids sir-12345678

# Create Spot Fleet
aws ec2 request-spot-fleet \\
    --spot-fleet-request-config file://spot-fleet-config.json

# spot-fleet-config.json
{
    "SpotPrice": "0.05",
    "TargetCapacity": 10,
    "IamFleetRole": "arn:aws:iam::123456789012:role/spot-fleet-role",
    "LaunchSpecifications": [
        {
            "ImageId": "ami-0c55b159cbfafe1f0",
            "InstanceType": "t2.micro",
            "KeyName": "my-key-pair",
            "SecurityGroups": [{"GroupId": "sg-12345678"}],
            "SubnetId": "subnet-12345678"
        },
        {
            "ImageId": "ami-0c55b159cbfafe1f0",
            "InstanceType": "t2.small",
            "KeyName": "my-key-pair",
            "SecurityGroups": [{"GroupId": "sg-12345678"}],
            "SubnetId": "subnet-87654321"
        }
    ],
    "AllocationStrategy": "diversified"
}

# Check interruption notice (on instance)
curl http://169.254.169.254/latest/meta-data/spot/instance-action

# Using boto3
import boto3
import json

ec2 = boto3.client('ec2')

# Request spot instance
response = ec2.request_spot_instances(
    SpotPrice='0.05',
    InstanceCount=1,
    Type='one-time',
    LaunchSpecification={
        'ImageId': 'ami-0c55b159cbfafe1f0',
        'InstanceType': 't2.micro',
        'KeyName': 'my-key-pair',
        'SecurityGroupIds': ['sg-12345678'],
        'SubnetId': 'subnet-12345678'
    }
)
spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

# Create Spot Fleet
spot_fleet_config = {
    'SpotPrice': '0.05',
    'TargetCapacity': 10,
    'IamFleetRole': 'arn:aws:iam::123456789012:role/spot-fleet-role',
    'LaunchSpecifications': [
        {
            'ImageId': 'ami-0c55b159cbfafe1f0',
            'InstanceType': 't2.micro',
            'KeyName': 'my-key-pair',
            'SecurityGroups': [{'GroupId': 'sg-12345678'}],
            'SubnetId': 'subnet-12345678'
        }
    ],
    'AllocationStrategy': 'diversified'
}

ec2.request_spot_fleet(SpotFleetRequestConfig=spot_fleet_config)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          303,
			Title:       "Amazon S3",
			Description: "Learn S3: buckets, objects, storage classes, versioning, lifecycle policies, and security.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "S3 Fundamentals",
					Content: `Amazon Simple Storage Service (S3) is object storage built to store and retrieve any amount of data.

**S3 Concepts:**
- **Bucket**: Container for objects (globally unique name)
- **Object**: File and metadata stored in bucket
- **Key**: Unique identifier for object in bucket
- **Region**: Geographic location of bucket
- **Storage Class**: Access tier (Standard, IA, Glacier, etc.)

**S3 Features:**
- **Durability**: 99.999999999% (11 9's)
- **Availability**: 99.99% for Standard
- **Scalability**: Unlimited storage
- **Security**: Encryption, access control, versioning
- **Lifecycle**: Automatic transitions and expiration

**Storage Classes:**
- **Standard**: General purpose, frequently accessed
- **Standard-IA**: Infrequent access, lower cost
- **One Zone-IA**: Single AZ, lower cost
- **Glacier Instant Retrieval**: Archive with instant access
- **Glacier Flexible Retrieval**: Archive (3 retrieval options)
- **Glacier Deep Archive**: Lowest cost, 12-hour retrieval
- **Intelligent-Tiering**: Automatic cost optimization

**Best Practices:**
- Use appropriate storage class
- Enable versioning for critical data
- Use lifecycle policies for cost optimization
- Enable encryption
- Use bucket policies for access control`,
					CodeExamples: `# Create S3 bucket
aws s3 mb s3://my-unique-bucket-name --region us-east-1

# Upload file to S3
aws s3 cp local-file.txt s3://my-bucket/path/to/file.txt

# Download file from S3
aws s3 cp s3://my-bucket/path/to/file.txt local-file.txt

# Sync directory to S3
aws s3 sync ./local-directory s3://my-bucket/prefix/

# List objects in bucket
aws s3 ls s3://my-bucket/

# Delete object
aws s3 rm s3://my-bucket/path/to/file.txt

# Delete bucket (must be empty)
aws s3 rb s3://my-bucket

# Using boto3
import boto3

s3 = boto3.client('s3')

# Create bucket
s3.create_bucket(
    Bucket='my-unique-bucket-name',
    CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
)

# Upload file
s3.upload_file('local-file.txt', 'my-bucket', 'remote-file.txt')

# Download file
s3.download_file('my-bucket', 'remote-file.txt', 'local-file.txt')

# List objects
response = s3.list_objects_v2(Bucket='my-bucket')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Delete object
s3.delete_object(Bucket='my-bucket', Key='remote-file.txt')`,
				},
				{
					Title: "S3 Versioning and Lifecycle",
					Content: `S3 versioning keeps multiple versions of objects, and lifecycle policies automate transitions and deletions.

**Versioning:**
- Keeps multiple versions of same object
- Protects against accidental deletion
- Can be enabled at bucket level
- Each version has unique version ID
- Can be suspended (not disabled)

**Lifecycle Policies:**
- Automate transitions between storage classes
- Automate object expiration
- Reduce storage costs
- Rules based on age or prefix

**Lifecycle Actions:**
- **Transition**: Move to different storage class
- **Expiration**: Delete objects after specified days
- **Abort incomplete multipart uploads**: Clean up failed uploads

**Use Cases:**
- Archive old data to Glacier
- Delete temporary files automatically
- Optimize costs with intelligent tiering
- Comply with retention policies

**Best Practices:**
- Enable versioning for production buckets
- Use lifecycle policies for cost optimization
- Set appropriate expiration dates
- Test lifecycle policies before applying
- Monitor lifecycle transitions`,
					CodeExamples: `# Enable versioning
aws s3api put-bucket-versioning \\
    --bucket my-bucket \\
    --versioning-configuration Status=Enabled

# Suspend versioning
aws s3api put-bucket-versioning \\
    --bucket my-bucket \\
    --versioning-configuration Status=Suspended

# List object versions
aws s3api list-object-versions \\
    --bucket my-bucket \\
    --prefix path/to/file.txt

# Delete specific version
aws s3api delete-object \\
    --bucket my-bucket \\
    --key path/to/file.txt \\
    --version-id version-id-here

# Create lifecycle policy
aws s3api put-bucket-lifecycle-configuration \\
    --bucket my-bucket \\
    --lifecycle-configuration file://lifecycle.json

# lifecycle.json
{
    "Rules": [
        {
            "Id": "TransitionToIA",
            "Status": "Enabled",
            "Prefix": "",
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                }
            ]
        },
        {
            "Id": "TransitionToGlacier",
            "Status": "Enabled",
            "Prefix": "archive/",
            "Transitions": [
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        },
        {
            "Id": "DeleteOldVersions",
            "Status": "Enabled",
            "Prefix": "",
            "NoncurrentVersionTransitions": [
                {
                    "NoncurrentDays": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "NoncurrentDays": 90,
                    "StorageClass": "GLACIER"
                }
            ],
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 365
            }
        },
        {
            "Id": "DeleteTempFiles",
            "Status": "Enabled",
            "Prefix": "temp/",
            "Expiration": {
                "Days": 7
            }
        }
    ]
}

# Using boto3
import boto3

s3 = boto3.client('s3')

# Enable versioning
s3.put_bucket_versioning(
    Bucket='my-bucket',
    VersioningConfiguration={'Status': 'Enabled'}
)

# Put lifecycle configuration
s3.put_bucket_lifecycle_configuration(
    Bucket='my-bucket',
    LifecycleConfiguration={
        'Rules': [
            {
                'Id': 'TransitionToIA',
                'Status': 'Enabled',
                'Transitions': [
                    {
                        'Days': 30,
                        'StorageClass': 'STANDARD_IA'
                    }
                ]
            }
        ]
    }
)`,
				},
				{
					Title: "S3 Encryption and Security",
					Content: `S3 provides multiple encryption options and security features to protect your data.

**Encryption Types:**
- **SSE-S3**: Server-side encryption with S3-managed keys
- **SSE-KMS**: Server-side encryption with KMS keys
- **SSE-C**: Server-side encryption with customer-provided keys
- **Client-Side Encryption**: Encrypt before uploading

**SSE-S3:**
- Default encryption for S3
- S3 manages keys
- AES-256 encryption
- No additional cost
- Automatic encryption

**SSE-KMS:**
- AWS KMS manages keys
- Customer Master Keys (CMKs)
- Audit key usage via CloudTrail
- Additional cost for KMS
- More control over keys

**Bucket Policies:**
- JSON policy documents
- Control access to buckets and objects
- Can allow/deny specific actions
- Support conditions (IP, time, etc.)
- Public access control

**Access Control:**
- **Bucket Policies**: Bucket-level permissions
- **ACLs**: Legacy access control (not recommended)
- **IAM Policies**: User/role permissions
- **Public Access Block**: Prevent public access

**Best Practices:**
- Enable encryption by default
- Use SSE-KMS for sensitive data
- Use bucket policies for access control
- Enable public access block
- Use IAM policies for fine-grained control
- Enable versioning for critical data
- Enable MFA Delete for production`,
					CodeExamples: `# Enable default encryption (SSE-S3)
aws s3api put-bucket-encryption \\
    --bucket my-bucket \\
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

# Enable encryption with KMS
aws s3api put-bucket-encryption \\
    --bucket my-bucket \\
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
            }
        }]
    }'

# Upload with encryption
aws s3 cp file.txt s3://my-bucket/file.txt \\
    --sse aws:kms \\
    --sse-kms-key-id alias/my-key

# Create bucket policy
aws s3api put-bucket-policy \\
    --bucket my-bucket \\
    --policy file://bucket-policy.json

# bucket-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        },
        {
            "Sid": "DenyPublicAccess",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::my-bucket/*",
            "Condition": {
                "Bool": {
                    "aws:ViaAWSService": "false"
                }
            }
        }
    ]
}

# Block public access
aws s3api put-public-access-block \\
    --bucket my-bucket \\
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Using boto3
import boto3

s3 = boto3.client('s3')

# Enable default encryption
s3.put_bucket_encryption(
    Bucket='my-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'AES256'
            }
        }]
    }
)

# Upload with KMS encryption
s3.upload_file(
    'file.txt',
    'my-bucket',
    'file.txt',
    ExtraArgs={
        'ServerSideEncryption': 'aws:kms',
        'SSEKMSKeyId': 'alias/my-key'
    }
)

# Put bucket policy
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::123456789012:user/john"},
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ]
        }
    ]
}

s3.put_bucket_policy(
    Bucket='my-bucket',
    Policy=json.dumps(bucket_policy)
)`,
				},
				{
					Title: "S3 Bucket Policies and CORS",
					Content: `Bucket policies control access to S3 buckets, and CORS enables cross-origin resource sharing.

**Bucket Policies:**
- JSON policy documents attached to buckets
- Control who can access what
- Support conditions and IP restrictions
- Can allow public access
- More flexible than ACLs

**Policy Elements:**
- **Effect**: Allow or Deny
- **Principal**: Who the policy applies to
- **Action**: What actions are allowed
- **Resource**: Which buckets/objects
- **Condition**: Optional conditions

**Common Use Cases:**
- Public website hosting
- Allow specific IAM users/roles
- IP-based restrictions
- Time-based access
- Require encryption

**CORS (Cross-Origin Resource Sharing):**
- Allows web pages to access S3 from different domains
- Required for browser-based uploads
- Configure allowed origins, methods, headers
- Set max age for preflight cache

**CORS Configuration:**
- **AllowedOrigins**: Domains that can access
- **AllowedMethods**: HTTP methods (GET, PUT, POST, etc.)
- **AllowedHeaders**: Headers allowed in requests
- **ExposedHeaders**: Headers exposed to client
- **MaxAgeSeconds**: Cache preflight requests

**Best Practices:**
- Use bucket policies over ACLs
- Follow principle of least privilege
- Test policies before applying
- Use conditions for additional security
- Configure CORS only when needed
- Limit CORS to specific origins`,
					CodeExamples: `# Bucket policy: Public read access
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}

# Bucket policy: IP restriction
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}

# Bucket policy: Require encryption
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::my-bucket/*",
            "Condition": {
                "StringNotEquals": {
                    "s3:x-amz-server-side-encryption": "AES256"
                }
            }
        }
    ]
}

# Configure CORS
aws s3api put-bucket-cors \\
    --bucket my-bucket \\
    --cors-configuration file://cors.json

# cors.json
{
    "CORSRules": [
        {
            "AllowedOrigins": ["https://example.com"],
            "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
            "AllowedHeaders": ["*"],
            "ExposedHeaders": ["ETag"],
            "MaxAgeSeconds": 3000
        },
        {
            "AllowedOrigins": ["https://*.example.com"],
            "AllowedMethods": ["GET"],
            "AllowedHeaders": ["Authorization"],
            "MaxAgeSeconds": 3600
        }
    ]
}

# Get CORS configuration
aws s3api get-bucket-cors --bucket my-bucket

# Delete CORS configuration
aws s3api delete-bucket-cors --bucket my-bucket

# Using boto3
import boto3
import json

s3 = boto3.client('s3')

# Put bucket policy
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::123456789012:user/john"},
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": "arn:aws:s3:::my-bucket/*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}

s3.put_bucket_policy(
    Bucket='my-bucket',
    Policy=json.dumps(bucket_policy)
)

# Configure CORS
cors_configuration = {
    'CORSRules': [
        {
            'AllowedOrigins': ['https://example.com'],
            'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
            'AllowedHeaders': ['*'],
            'ExposedHeaders': ['ETag'],
            'MaxAgeSeconds': 3000
        }
    ]
}

s3.put_bucket_cors(
    Bucket='my-bucket',
    CORSConfiguration=cors_configuration
)`,
				},
				{
					Title: "S3 Event Notifications and Lambda Triggers",
					Content: `S3 can send event notifications when objects are created, deleted, or modified, enabling event-driven architectures.

**Event Notification Types:**
- **s3:ObjectCreated**: Object created (Put, Post, Copy, CompleteMultipartUpload)
- **s3:ObjectRemoved**: Object deleted (Delete, DeleteMarkerCreated)
- **s3:ObjectRestore**: Glacier restore initiated/completed
- **s3:ReducedRedundancyLostObject**: RRS object lost

**Event Destinations:**
- **SQS Queue**: Simple queue service
- **SNS Topic**: Simple notification service
- **Lambda Function**: Serverless function
- **EventBridge**: Event bus (newer)

**Event Structure:**
- Event name (e.g., ObjectCreated:Put)
- Bucket name
- Object key
- Event time
- Request ID
- User identity

**Filtering:**
- Filter by prefix
- Filter by suffix
- Filter by object tags
- Multiple filters per notification

**Use Cases:**
- Process uploaded images
- Trigger data pipelines
- Send notifications
- Audit object access
- Automate workflows
- Index objects in search

**Best Practices:**
- Use appropriate destination
- Filter events to reduce noise
- Handle errors gracefully
- Monitor event delivery
- Use EventBridge for complex routing
- Test event handlers`,
					CodeExamples: `# Configure S3 event notification to Lambda
aws s3api put-bucket-notification-configuration \\
    --bucket my-bucket \\
    --notification-configuration file://notification.json

# notification.json
{
    "LambdaFunctionConfigurations": [
        {
            "LambdaFunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:process-image",
            "Events": ["s3:ObjectCreated:Put"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {
                            "Name": "prefix",
                            "Value": "uploads/images/"
                        },
                        {
                            "Name": "suffix",
                            "Value": ".jpg"
                        }
                    ]
                }
            }
        }
    ]
}

# Configure S3 event to SQS
{
    "QueueConfigurations": [
        {
            "QueueArn": "arn:aws:sqs:us-east-1:123456789012:my-queue",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {
                            "Name": "prefix",
                            "Value": "uploads/"
                        }
                    ]
                }
            }
        }
    ]
}

# Configure S3 event to SNS
{
    "TopicConfigurations": [
        {
            "TopicArn": "arn:aws:sns:us-east-1:123456789012:my-topic",
            "Events": ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
        }
    ]
}

# Configure S3 event to EventBridge
aws s3api put-bucket-notification-configuration \\
    --bucket my-bucket \\
    --notification-configuration '{
        "EventBridgeConfiguration": {}
    }'

# Lambda function to process S3 events
import json
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        event_name = record['eventName']
        
        print(f"Event: {event_name}")
        print(f"Bucket: {bucket}")
        print(f"Key: {key}")
        
        # Process object
        if event_name.startswith('ObjectCreated'):
            # Download and process
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            # Process content...
            
    return {'statusCode': 200}

# Using boto3
import boto3

s3 = boto3.client('s3')

# Configure notification
notification_config = {
    'LambdaFunctionConfigurations': [
        {
            'LambdaFunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:process-image',
            'Events': ['s3:ObjectCreated:Put'],
            'Filter': {
                'Key': {
                    'FilterRules': [
                        {'Name': 'prefix', 'Value': 'uploads/images/'},
                        {'Name': 'suffix', 'Value': '.jpg'}
                    ]
                }
            }
        }
    ]
}

s3.put_bucket_notification_configuration(
    Bucket='my-bucket',
    NotificationConfiguration=notification_config
)`,
				},
				{
					Title: "S3 Transfer Acceleration and Multipart Uploads",
					Content: `S3 Transfer Acceleration speeds up uploads, and multipart uploads enable efficient large file transfers.

**Transfer Acceleration:**
- Uses CloudFront edge locations
- Faster uploads from distant locations
- Uses optimized network path
- Additional cost per GB transferred
- Enable per bucket

**How It Works:**
- Upload to edge location
- Edge location forwards to S3
- Optimized AWS backbone network
- Reduces latency for distant users

**Use Cases:**
- Uploads from distant locations
- Large file uploads
- Global applications
- Better user experience

**Multipart Upload:**
- Split large files into parts
- Upload parts in parallel
- More reliable for large files
- Can resume failed uploads
- Recommended for files > 100MB

**Multipart Upload Process:**
1. Initiate multipart upload
2. Upload parts (can be parallel)
3. Complete multipart upload
4. S3 assembles parts into object

**Part Management:**
- Parts can be 5MB to 5GB (except last)
- Up to 10,000 parts
- Can abort incomplete uploads
- List parts before completing

**Best Practices:**
- Use Transfer Acceleration for distant uploads
- Use multipart for files > 100MB
- Upload parts in parallel
- Clean up incomplete uploads
- Monitor transfer performance`,
					CodeExamples: `# Enable Transfer Acceleration
aws s3api put-bucket-accelerate-configuration \\
    --bucket my-bucket \\
    --accelerate-configuration Status=Enabled

# Check Transfer Acceleration status
aws s3api get-bucket-accelerate-configuration --bucket my-bucket

# Upload with Transfer Acceleration
aws s3 cp large-file.zip s3://my-bucket/large-file.zip \\
    --endpoint-url https://s3-accelerate.amazonaws.com

# Initiate multipart upload
aws s3api create-multipart-upload \\
    --bucket my-bucket \\
    --key large-file.zip \\
    --metadata '{"original-name": "large-file.zip"}'

# Upload part
aws s3api upload-part \\
    --bucket my-bucket \\
    --key large-file.zip \\
    --part-number 1 \\
    --upload-id upload-id-here \\
    --body part1.bin

# List uploaded parts
aws s3api list-parts \\
    --bucket my-bucket \\
    --key large-file.zip \\
    --upload-id upload-id-here

# Complete multipart upload
aws s3api complete-multipart-upload \\
    --bucket my-bucket \\
    --key large-file.zip \\
    --upload-id upload-id-here \\
    --multipart-upload file://parts.json

# parts.json
{
    "Parts": [
        {
            "ETag": "etag-part-1",
            "PartNumber": 1
        },
        {
            "ETag": "etag-part-2",
            "PartNumber": 2
        }
    ]
}

# Abort multipart upload
aws s3api abort-multipart-upload \\
    --bucket my-bucket \\
    --key large-file.zip \\
    --upload-id upload-id-here

# Using boto3
import boto3
import os

s3 = boto3.client('s3')

# Enable Transfer Acceleration
s3.put_bucket_accelerate_configuration(
    Bucket='my-bucket',
    AccelerateConfiguration={'Status': 'Enabled'}
)

# Upload with Transfer Acceleration
s3_transfer_accel = boto3.client('s3', endpoint_url='https://s3-accelerate.amazonaws.com')
s3_transfer_accel.upload_file('large-file.zip', 'my-bucket', 'large-file.zip')

# Multipart upload
def upload_large_file(bucket, key, filepath):
    # Initiate multipart upload
    response = s3.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id = response['UploadId']
    
    try:
        parts = []
        file_size = os.path.getsize(filepath)
        part_size = 5 * 1024 * 1024  # 5MB
        
        with open(filepath, 'rb') as f:
            part_num = 1
            while f.tell() < file_size:
                part_data = f.read(part_size)
                part_response = s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    PartNumber=part_num,
                    UploadId=upload_id,
                    Body=part_data
                )
                parts.append({
                    'ETag': part_response['ETag'],
                    'PartNumber': part_num
                })
                part_num += 1
        
        # Complete multipart upload
        s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )
    except Exception as e:
        # Abort on error
        s3.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id
        )
        raise e`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          304,
			Title:       "Amazon VPC Basics",
			Description: "Introduction to VPC: virtual private clouds, subnets, route tables, internet gateways, and basic networking.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "VPC Fundamentals",
					Content: `Amazon Virtual Private Cloud (VPC) lets you provision a logically isolated section of the AWS cloud.

**VPC Concepts:**
- **VPC**: Isolated virtual network in AWS
- **Subnet**: Range of IP addresses in VPC
- **Route Table**: Rules for routing network traffic
- **Internet Gateway**: Allows internet access
- **NAT Gateway**: Allows private subnet internet access
- **Security Groups**: Instance-level firewall
- **Network ACLs**: Subnet-level firewall

**Default VPC:**
- Created automatically in each region
- Internet gateway attached
- Public subnets in each AZ
- Default security group
- Default route table

**Custom VPC:**
- Full control over network configuration
- Choose IP address range (CIDR)
- Create subnets in specific AZs
- Configure route tables
- Set up internet and NAT gateways

**IP Addressing:**
- CIDR blocks define IP ranges
- /16 provides 65,536 IPs
- /24 provides 256 IPs
- Cannot overlap with existing networks

**Best Practices:**
- Use private subnets for resources
- Use public subnets for load balancers
- Implement least privilege security
- Use multiple AZs for high availability
- Document network architecture`,
					CodeExamples: `# Create VPC
aws ec2 create-vpc \\
    --cidr-block 10.0.0.0/16 \\
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# Create subnet
aws ec2 create-subnet \\
    --vpc-id vpc-12345678 \\
    --cidr-block 10.0.1.0/24 \\
    --availability-zone us-east-1a \\
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=PublicSubnet1}]'

# Create internet gateway
aws ec2 create-internet-gateway \\
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MyIGW}]'

# Attach internet gateway to VPC
aws ec2 attach-internet-gateway \\
    --internet-gateway-id igw-12345678 \\
    --vpc-id vpc-12345678

# Create route table
aws ec2 create-route-table \\
    --vpc-id vpc-12345678 \\
    --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=PublicRouteTable}]'

# Add route to internet gateway
aws ec2 create-route \\
    --route-table-id rtb-12345678 \\
    --destination-cidr-block 0.0.0.0/0 \\
    --gateway-id igw-12345678

# Associate subnet with route table
aws ec2 associate-route-table \\
    --subnet-id subnet-12345678 \\
    --route-table-id rtb-12345678

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create VPC
vpc_response = ec2.create_vpc(CidrBlock='10.0.0.0/16')
vpc_id = vpc_response['Vpc']['VpcId']

# Create subnet
subnet_response = ec2.create_subnet(
    VpcId=vpc_id,
    CidrBlock='10.0.1.0/24',
    AvailabilityZone='us-east-1a'
)
subnet_id = subnet_response['Subnet']['SubnetId']

# Create internet gateway
igw_response = ec2.create_internet_gateway()
igw_id = igw_response['InternetGateway']['InternetGatewayId']

# Attach IGW to VPC
ec2.attach_internet_gateway(
    InternetGatewayId=igw_id,
    VpcId=vpc_id
)

# Create route table
rt_response = ec2.create_route_table(VpcId=vpc_id)
rt_id = rt_response['RouteTable']['RouteTableId']

# Add route
ec2.create_route(
    RouteTableId=rt_id,
    DestinationCidrBlock='0.0.0.0/0',
    GatewayId=igw_id
)

# Associate subnet
ec2.associate_route_table(
    SubnetId=subnet_id,
    RouteTableId=rt_id
)`,
				},
				{
					Title: "NAT Gateways and NAT Instances",
					Content: `NAT Gateways allow instances in private subnets to access the internet while remaining private.

**NAT Gateway:**
- Managed service by AWS
- Highly available (in single AZ)
- Scales automatically
- No maintenance required
- Charges per hour and data processed
- Use multiple NAT Gateways for multi-AZ

**NAT Instance:**
- EC2 instance running NAT software
- Manual management required
- Can be used for custom configurations
- Less reliable than NAT Gateway
- Not recommended for production

**How NAT Works:**
- Private subnet instances route to NAT Gateway
- NAT Gateway translates private IP to public IP
- Outbound internet access allowed
- Inbound connections blocked
- Return traffic routed back correctly

**Use Cases:**
- Private subnet instances need internet access
- Download updates and patches
- Access AWS APIs
- Outbound connections only

**Best Practices:**
- Use NAT Gateway over NAT Instance
- Deploy NAT Gateway in public subnet
- Use multiple NAT Gateways for high availability
- Monitor NAT Gateway metrics
- Use VPC endpoints to reduce NAT costs`,
					CodeExamples: `# Allocate Elastic IP for NAT Gateway
aws ec2 allocate-address --domain vpc

# Create NAT Gateway
aws ec2 create-nat-gateway \\
    --subnet-id subnet-12345678 \\
    --allocation-id eipalloc-12345678 \\
    --tag-specifications 'ResourceType=nat-gateway,Tags=[{Key=Name,Value=MyNATGateway}]'

# Create route table for private subnet
aws ec2 create-route-table \\
    --vpc-id vpc-12345678 \\
    --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=PrivateRouteTable}]'

# Add route to NAT Gateway
aws ec2 create-route \\
    --route-table-id rtb-87654321 \\
    --destination-cidr-block 0.0.0.0/0 \\
    --nat-gateway-id nat-1234567890abcdef0

# Associate private subnet with route table
aws ec2 associate-route-table \\
    --subnet-id subnet-87654321 \\
    --route-table-id rtb-87654321

# Describe NAT Gateway
aws ec2 describe-nat-gateways \\
    --nat-gateway-ids nat-1234567890abcdef0

# Delete NAT Gateway
aws ec2 delete-nat-gateway --nat-gateway-id nat-1234567890abcdef0

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Allocate Elastic IP
eip_response = ec2.allocate_address(Domain='vpc')
allocation_id = eip_response['AllocationId']

# Create NAT Gateway
nat_response = ec2.create_nat_gateway(
    SubnetId='subnet-12345678',
    AllocationId=allocation_id
)
nat_gateway_id = nat_response['NatGateway']['NatGatewayId']

# Wait for NAT Gateway to be available
waiter = ec2.get_waiter('nat_gateway_available')
waiter.wait(NatGatewayIds=[nat_gateway_id])

# Create private route table
rt_response = ec2.create_route_table(VpcId='vpc-12345678')
private_rt_id = rt_response['RouteTable']['RouteTableId']

# Add route to NAT Gateway
ec2.create_route(
    RouteTableId=private_rt_id,
    DestinationCidrBlock='0.0.0.0/0',
    NatGatewayId=nat_gateway_id
)

# Associate private subnet
ec2.associate_route_table(
    SubnetId='subnet-87654321',
    RouteTableId=private_rt_id
)`,
				},
				{
					Title: "VPC Endpoints",
					Content: `VPC Endpoints allow private connectivity to AWS services without using internet gateway or NAT.

**VPC Endpoint Types:**
- **Gateway Endpoints**: For S3 and DynamoDB (free)
- **Interface Endpoints**: For other AWS services (charges apply)
- **Gateway Load Balancer Endpoints**: For third-party security services

**Gateway Endpoints:**
- Route table entries only
- No ENI or security groups
- Free of charge
- Only for S3 and DynamoDB
- Automatic route propagation

**Interface Endpoints:**
- ENI in your subnet
- Uses PrivateLink
- Charges per hour and data
- Security groups apply
- DNS names provided
- High availability

**Benefits:**
- Keep traffic within AWS network
- No internet gateway needed
- Improved security
- Lower latency
- Reduced NAT Gateway costs

**Use Cases:**
- Access S3 from private subnets
- Access DynamoDB privately
- Access other AWS services privately
- Compliance requirements
- Cost optimization

**Best Practices:**
- Use Gateway Endpoints for S3/DynamoDB
- Use Interface Endpoints for other services
- Deploy in multiple AZs for HA
- Configure security groups properly
- Use endpoint policies for access control`,
					CodeExamples: `# Create Gateway Endpoint for S3
aws ec2 create-vpc-endpoint \\
    --vpc-id vpc-12345678 \\
    --service-name com.amazonaws.us-east-1.s3 \\
    --route-table-ids rtb-12345678 rtb-87654321 \\
    --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=S3Endpoint}]'

# Create Gateway Endpoint for DynamoDB
aws ec2 create-vpc-endpoint \\
    --vpc-id vpc-12345678 \\
    --service-name com.amazonaws.us-east-1.dynamodb \\
    --route-table-ids rtb-12345678 \\
    --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=DynamoDBEndpoint}]'

# Create Interface Endpoint for Lambda
aws ec2 create-vpc-endpoint \\
    --vpc-id vpc-12345678 \\
    --service-name com.amazonaws.us-east-1.lambda \\
    --vpc-endpoint-type Interface \\
    --subnet-ids subnet-12345678 subnet-87654321 \\
    --security-group-ids sg-12345678 \\
    --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=LambdaEndpoint}]'

# Create Interface Endpoint with endpoint policy
aws ec2 create-vpc-endpoint \\
    --vpc-id vpc-12345678 \\
    --service-name com.amazonaws.us-east-1.s3 \\
    --vpc-endpoint-type Interface \\
    --subnet-ids subnet-12345678 \\
    --security-group-ids sg-12345678 \\
    --policy-document file://endpoint-policy.json

# endpoint-policy.json
{
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}

# Describe VPC endpoints
aws ec2 describe-vpc-endpoints

# Delete VPC endpoint
aws ec2 delete-vpc-endpoint --vpc-endpoint-id vpce-1234567890abcdef0

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create Gateway Endpoint for S3
s3_endpoint = ec2.create_vpc_endpoint(
    VpcId='vpc-12345678',
    ServiceName='com.amazonaws.us-east-1.s3',
    RouteTableIds=['rtb-12345678', 'rtb-87654321'],
    VpcEndpointType='Gateway'
)
print(f"S3 Endpoint ID: {s3_endpoint['VpcEndpoint']['VpcEndpointId']}")

# Create Interface Endpoint for SNS
sns_endpoint = ec2.create_vpc_endpoint(
    VpcId='vpc-12345678',
    ServiceName='com.amazonaws.us-east-1.sns',
    VpcEndpointType='Interface',
    SubnetIds=['subnet-12345678', 'subnet-87654321'],
    SecurityGroupIds=['sg-12345678']
)
print(f"SNS Endpoint ID: {sns_endpoint['VpcEndpoint']['VpcEndpointId']}")

# List all endpoints
endpoints = ec2.describe_vpc_endpoints()
for endpoint in endpoints['VpcEndpoints']:
    print(f"Endpoint: {endpoint['VpcEndpointId']}, Service: {endpoint['ServiceName']}")`,
				},
				{
					Title: "Network ACLs",
					Content: `Network ACLs (NACLs) provide subnet-level firewall rules for controlling traffic.

**Network ACLs:**
- Stateless: Must allow both inbound and outbound
- Subnet-level firewall
- Rules evaluated in order
- Default allow all (can be changed)
- Free of charge

**Security Groups vs NACLs:**
- **Security Groups**: Instance-level, stateful, allow rules only
- **NACLs**: Subnet-level, stateless, allow and deny rules

**NACL Rules:**
- Rule number (1-32766)
- Protocol (TCP, UDP, ICMP, etc.)
- Port range
- Source/Destination CIDR
- Allow or Deny
- Evaluated in order (lowest to highest)

**Default NACL:**
- Allows all inbound and outbound traffic
- Can be modified
- Associated with all subnets by default

**Custom NACL:**
- Denies all traffic by default
- Must explicitly allow traffic
- Can be associated with subnets

**Best Practices:**
- Use NACLs for subnet-level rules
- Use Security Groups for instance-level rules
- Deny specific IPs/ranges with NACLs
- Test rules before applying
- Document rule purposes
- Use rule numbers with gaps for future rules`,
					CodeExamples: `# Create custom Network ACL
aws ec2 create-network-acl \\
    --vpc-id vpc-12345678 \\
    --tag-specifications 'ResourceType=network-acl,Tags=[{Key=Name,Value=PrivateNACL}]'

# Create inbound rule (allow HTTP)
aws ec2 create-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 100 \\
    --protocol tcp \\
    --port-range From=80,To=80 \\
    --cidr-block 0.0.0.0/0 \\
    --egress false \\
    --rule-action allow

# Create inbound rule (allow HTTPS)
aws ec2 create-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 110 \\
    --protocol tcp \\
    --port-range From=443,To=443 \\
    --cidr-block 0.0.0.0/0 \\
    --egress false \\
    --rule-action allow

# Create inbound rule (allow SSH from specific IP)
aws ec2 create-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 120 \\
    --protocol tcp \\
    --port-range From=22,To=22 \\
    --cidr-block 203.0.113.0/24 \\
    --egress false \\
    --rule-action allow

# Create outbound rule (allow all)
aws ec2 create-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 100 \\
    --protocol -1 \\
    --cidr-block 0.0.0.0/0 \\
    --egress true \\
    --rule-action allow

# Deny specific IP range
aws ec2 create-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 200 \\
    --protocol -1 \\
    --cidr-block 192.0.2.0/24 \\
    --egress false \\
    --rule-action deny

# Associate NACL with subnet
aws ec2 replace-network-acl-association \\
    --association-id aclassoc-12345678 \\
    --network-acl-id acl-12345678

# Describe Network ACLs
aws ec2 describe-network-acls \\
    --network-acl-ids acl-12345678

# Delete Network ACL entry
aws ec2 delete-network-acl-entry \\
    --network-acl-id acl-12345678 \\
    --rule-number 200 \\
    --egress false

# Using boto3
import boto3

ec2 = boto3.client('ec2')

# Create custom NACL
nacl_response = ec2.create_network_acl(
    VpcId='vpc-12345678'
)
nacl_id = nacl_response['NetworkAcl']['NetworkAclId']

# Create inbound rules
ec2.create_network_acl_entry(
    NetworkAclId=nacl_id,
    RuleNumber=100,
    Protocol='tcp',
    PortRange={'From': 80, 'To': 80},
    CidrBlock='0.0.0.0/0',
    Egress=False,
    RuleAction='allow'
)

ec2.create_network_acl_entry(
    NetworkAclId=nacl_id,
    RuleNumber=110,
    Protocol='tcp',
    PortRange={'From': 443, 'To': 443},
    CidrBlock='0.0.0.0/0',
    Egress=False,
    RuleAction='allow'
)

# Create outbound rule
ec2.create_network_acl_entry(
    NetworkAclId=nacl_id,
    RuleNumber=100,
    Protocol='-1',  # All protocols
    CidrBlock='0.0.0.0/0',
    Egress=True,
    RuleAction='allow'
)

# Associate with subnet
# First get current association
associations = ec2.describe_network_acls(
    NetworkAclIds=[nacl_id]
)['NetworkAcls'][0]['Associations']

# Replace association
for assoc in associations:
    if assoc['SubnetId'] == 'subnet-12345678':
        ec2.replace_network_acl_association(
            AssociationId=assoc['NetworkAclAssociationId'],
            NetworkAclId=nacl_id
        )`,
				},
				{
					Title: "VPC Flow Logs",
					Content: `VPC Flow Logs capture information about IP traffic going to and from network interfaces in your VPC.

**Flow Log Types:**
- **VPC Flow Logs**: All traffic in VPC
- **Subnet Flow Logs**: Traffic in subnet
- **ENI Flow Logs**: Traffic for specific network interface

**Flow Log Destinations:**
- CloudWatch Logs
- S3 bucket
- Cannot send to both simultaneously

**Flow Log Format:**
- Version
- Account ID
- Interface ID
- Source/Destination IP
- Source/Destination Port
- Protocol
- Packets
- Bytes
- Start/End time
- Action (ACCEPT/REJECT)
- Log status

**Use Cases:**
- Troubleshoot connectivity issues
- Security monitoring
- Network analysis
- Compliance requirements
- Cost optimization

**Limitations:**
- Cannot enable for peered VPCs
- Cannot tag flow logs
- Does not capture all traffic (DNS, DHCP, etc.)
- Charges apply for CloudWatch Logs/S3

**Best Practices:**
- Enable flow logs for production VPCs
- Send to S3 for long-term storage
- Use CloudWatch Logs for real-time analysis
- Analyze logs regularly
- Set up alerts for anomalies`,
					CodeExamples: `# Create IAM role for VPC Flow Logs
aws iam create-role \\
    --role-name vpc-flow-logs-role \\
    --assume-role-policy-document file://trust-policy.json

# trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "vpc-flow-logs.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

# Attach policy for CloudWatch Logs
aws iam attach-role-policy \\
    --role-name vpc-flow-logs-role \\
    --policy-arn arn:aws:iam::aws:policy/service-role/VPCFlowLogsRole

# Create CloudWatch Log Group
aws logs create-log-group --log-group-name VPCFlowLogs

# Create VPC Flow Log (CloudWatch Logs)
aws ec2 create-flow-logs \\
    --resource-type VPC \\
    --resource-ids vpc-12345678 \\
    --traffic-type ALL \\
    --log-destination-type cloud-watch-logs \\
    --log-destination arn:aws:logs:us-east-1:123456789012:log-group:VPCFlowLogs \\
    --deliver-logs-permission-arn arn:aws:iam::123456789012:role/vpc-flow-logs-role

# Create VPC Flow Log (S3)
aws ec2 create-flow-logs \\
    --resource-type VPC \\
    --resource-ids vpc-12345678 \\
    --traffic-type ALL \\
    --log-destination-type s3 \\
    --log-destination arn:aws:s3:::my-flow-logs-bucket/flow-logs/

# Create Subnet Flow Log
aws ec2 create-flow-logs \\
    --resource-type Subnet \\
    --resource-ids subnet-12345678 \\
    --traffic-type ALL \\
    --log-destination-type s3 \\
    --log-destination arn:aws:s3:::my-flow-logs-bucket/flow-logs/

# Create ENI Flow Log
aws ec2 create-flow-logs \\
    --resource-type NetworkInterface \\
    --resource-ids eni-12345678 \\
    --traffic-type ALL \\
    --log-destination-type s3 \\
    --log-destination arn:aws:s3:::my-flow-logs-bucket/flow-logs/

# Describe Flow Logs
aws ec2 describe-flow-logs

# Delete Flow Log
aws ec2 delete-flow-logs --flow-log-ids fl-1234567890abcdef0

# Query Flow Logs in CloudWatch Logs Insights
# Example query: Find rejected traffic
fields @timestamp, srcaddr, dstaddr, srcport, dstport, protocol, action
| filter action = "REJECT"
| sort @timestamp desc
| limit 100

# Using boto3
import boto3

ec2 = boto3.client('ec2')
logs = boto3.client('logs')

# Create log group
logs.create_log_group(logGroupName='VPCFlowLogs')

# Create VPC Flow Log
flow_log = ec2.create_flow_logs(
    ResourceType='VPC',
    ResourceIds=['vpc-12345678'],
    TrafficType='ALL',
    LogDestinationType='cloud-watch-logs',
    LogDestination='arn:aws:logs:us-east-1:123456789012:log-group:VPCFlowLogs',
    DeliverLogsPermissionArn='arn:aws:iam::123456789012:role/vpc-flow-logs-role'
)
print(f"Flow Log ID: {flow_log['FlowLogIds'][0]}")

# Create Flow Log to S3
s3_flow_log = ec2.create_flow_logs(
    ResourceType='VPC',
    ResourceIds=['vpc-12345678'],
    TrafficType='ALL',
    LogDestinationType='s3',
    LogDestination='arn:aws:s3:::my-flow-logs-bucket/flow-logs/'
)

# Describe Flow Logs
flow_logs = ec2.describe_flow_logs()
for fl in flow_logs['FlowLogs']:
    print(f"Flow Log: {fl['FlowLogId']}, Resource: {fl['ResourceId']}, Status: {fl['FlowLogStatus']}")

# Delete Flow Log
ec2.delete_flow_logs(FlowLogIds=['fl-1234567890abcdef0'])`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          305,
			Title:       "Amazon DynamoDB",
			Description: "Learn DynamoDB: NoSQL database, tables, items, indexes, streams, and best practices.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "DynamoDB Fundamentals",
					Content: `Amazon DynamoDB is a fully managed NoSQL database service providing fast and predictable performance.

**DynamoDB Concepts:**
- **Table**: Collection of items
- **Item**: Collection of attributes (like a row)
- **Attribute**: Name-value pair (like a column)
- **Primary Key**: Uniquely identifies each item
- **Partition Key**: Simple primary key
- **Composite Key**: Partition key + sort key

**Key Features:**
- Fully managed (no servers)
- Automatic scaling
- Built-in security
- Backup and restore
- Global tables
- Point-in-time recovery

**Read/Write Capacity:**
- **Provisioned**: Set read/write capacity units
- **On-Demand**: Pay per request (automatic scaling)
- **Auto Scaling**: Automatically adjust capacity

**Data Types:**
- Scalar: String, Number, Binary, Boolean, Null
- Document: List, Map
- Set: String Set, Number Set, Binary Set

**Best Practices:**
- Design for single-table access patterns
- Use appropriate primary key design
- Use indexes for query flexibility
- Monitor performance metrics
- Use on-demand for variable workloads`,
					CodeExamples: `# Create table with partition key
aws dynamodb create-table \\
    --table-name Users \\
    --attribute-definitions AttributeName=UserId,AttributeType=S \\
    --key-schema AttributeName=UserId,KeyType=HASH \\
    --billing-mode PROVISIONED \\
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5

# Create table with composite key
aws dynamodb create-table \\
    --table-name Orders \\
    --attribute-definitions \\
        AttributeName=CustomerId,AttributeType=S \\
        AttributeName=OrderId,AttributeType=S \\
    --key-schema \\
        AttributeName=CustomerId,KeyType=HASH \\
        AttributeName=OrderId,KeyType=RANGE \\
    --billing-mode ON_DEMAND

# Put item
aws dynamodb put-item \\
    --table-name Users \\
    --item '{
        "UserId": {"S": "user123"},
        "Name": {"S": "John Doe"},
        "Email": {"S": "john@example.com"},
        "Age": {"N": "30"}
    }'

# Get item
aws dynamodb get-item \\
    --table-name Users \\
    --key '{"UserId": {"S": "user123"}}'

# Query table
aws dynamodb query \\
    --table-name Orders \\
    --key-condition-expression "CustomerId = :cid" \\
    --expression-attribute-values '{":cid": {"S": "customer123"}}'

# Scan table
aws dynamodb scan \\
    --table-name Users \\
    --filter-expression "Age > :age" \\
    --expression-attribute-values '{":age": {"N": "25"}}'

# Using boto3
import boto3

dynamodb = boto3.resource('dynamodb')

# Create table
table = dynamodb.create_table(
    TableName='Users',
    KeySchema=[
        {'AttributeName': 'UserId', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'UserId', 'AttributeType': 'S'}
    ],
    BillingMode='PROVISIONED',
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

# Wait for table to be created
table.wait_until_exists()

# Put item
table.put_item(
    Item={
        'UserId': 'user123',
        'Name': 'John Doe',
        'Email': 'john@example.com',
        'Age': 30
    }
)

# Get item
response = table.get_item(Key={'UserId': 'user123'})
item = response.get('Item')
print(item)`,
				},
				{
					Title: "DynamoDB Indexes",
					Content: `Indexes enable efficient queries on attributes other than the primary key.

**Index Types:**
- **Global Secondary Index (GSI)**: Can have different partition and sort keys
- **Local Secondary Index (LSI)**: Same partition key, different sort key

**Global Secondary Index:**
- Can query any attribute
- Separate partition and sort keys
- Eventually consistent (can be strongly consistent)
- Can be created at any time
- Charges apply for storage and throughput

**Local Secondary Index:**
- Same partition key as base table
- Different sort key
- Strongly consistent
- Must be created with table
- Shares throughput with base table

**Use Cases:**
- Query by different attributes
- Support multiple access patterns
- Improve query performance
- Enable sorting by different attributes

**Best Practices:**
- Design indexes based on access patterns
- Use sparse indexes to save space
- Monitor index usage
- Right-size index capacity
- Use projection to minimize storage`,
					CodeExamples: `# Create table with GSI
aws dynamodb create-table \\
    --table-name Products \\
    --attribute-definitions \\
        AttributeName=ProductId,AttributeType=S \\
        AttributeName=Category,AttributeType=S \\
        AttributeName=Price,AttributeType=N \\
    --key-schema \\
        AttributeName=ProductId,KeyType=HASH \\
    --global-secondary-indexes '[
        {
            "IndexName": "CategoryIndex",
            "KeySchema": [
                {"AttributeName": "Category", "KeyType": "HASH"},
                {"AttributeName": "Price", "KeyType": "RANGE"}
            ],
            "Projection": {"ProjectionType": "ALL"},
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5
            }
        }
    ]' \\
    --billing-mode PROVISIONED \\
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5

# Query GSI
aws dynamodb query \\
    --table-name Products \\
    --index-name CategoryIndex \\
    --key-condition-expression "Category = :cat AND Price > :price" \\
    --expression-attribute-values '{":cat": {"S": "Electronics"}, ":price": {"N": "100"}}'

# Using boto3
import boto3

dynamodb = boto3.resource('dynamodb')

# Create table with GSI
table = dynamodb.create_table(
    TableName='Products',
    KeySchema=[
        {'AttributeName': 'ProductId', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'ProductId', 'AttributeType': 'S'},
        {'AttributeName': 'Category', 'AttributeType': 'S'},
        {'AttributeName': 'Price', 'AttributeType': 'N'}
    ],
    GlobalSecondaryIndexes=[
        {
            'IndexName': 'CategoryIndex',
            'KeySchema': [
                {'AttributeName': 'Category', 'KeyType': 'HASH'},
                {'AttributeName': 'Price', 'KeyType': 'RANGE'}
            ],
            'Projection': {'ProjectionType': 'ALL'},
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        }
    ],
    BillingMode='PROVISIONED',
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

# Query using GSI
table = dynamodb.Table('Products')
response = table.query(
    IndexName='CategoryIndex',
    KeyConditionExpression='Category = :cat AND Price > :price',
    ExpressionAttributeValues={
        ':cat': 'Electronics',
        ':price': 100
    }
)
for item in response['Items']:
    print(item)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          306,
			Title:       "Amazon CloudWatch",
			Description: "Learn CloudWatch: monitoring, metrics, alarms, logs, dashboards, and observability.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "CloudWatch Metrics and Alarms",
					Content: `CloudWatch collects and tracks metrics, and alarms notify you when metrics exceed thresholds.

**CloudWatch Metrics:**
- Data points over time
- Namespace, metric name, dimensions
- Automatic metrics from AWS services
- Custom metrics from applications
- Stored for 15 months

**Metric Types:**
- **Standard Resolution**: 1-minute granularity
- **High Resolution**: 1-second granularity
- **Custom Metrics**: Your application metrics

**CloudWatch Alarms:**
- Monitor metrics and trigger actions
- Send notifications (SNS)
- Auto Scaling actions
- EC2 actions (stop, terminate, recover)
- Multiple thresholds and periods

**Alarm States:**
- **OK**: Within threshold
- **ALARM**: Exceeded threshold
- **INSUFFICIENT_DATA**: Not enough data

**Best Practices:**
- Set up alarms for critical metrics
- Use appropriate evaluation periods
- Combine multiple metrics
- Test alarm actions
- Monitor alarm state changes`,
					CodeExamples: `# Put custom metric
aws cloudwatch put-metric-data \\
    --namespace MyApp \\
    --metric-name RequestCount \\
    --value 100 \\
    --unit Count

# Create alarm
aws cloudwatch put-metric-alarm \\
    --alarm-name high-cpu \\
    --alarm-description "Alert when CPU exceeds 80%" \\
    --metric-name CPUUtilization \\
    --namespace AWS/EC2 \\
    --statistic Average \\
    --period 300 \\
    --threshold 80 \\
    --comparison-operator GreaterThanThreshold \\
    --evaluation-periods 2 \\
    --alarm-actions arn:aws:sns:us-east-1:123456789012:alerts

# Get metric statistics
aws cloudwatch get-metric-statistics \\
    --namespace AWS/EC2 \\
    --metric-name CPUUtilization \\
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \\
    --start-time 2024-01-01T00:00:00Z \\
    --end-time 2024-01-01T23:59:59Z \\
    --period 3600 \\
    --statistics Average,Maximum

# Using boto3
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Put custom metric
cloudwatch.put_metric_data(
    Namespace='MyApp',
    MetricData=[
        {
            'MetricName': 'RequestCount',
            'Value': 100,
            'Unit': 'Count'
        }
    ]
)

# Create alarm
cloudwatch.put_metric_alarm(
    AlarmName='high-cpu',
    AlarmDescription='Alert when CPU exceeds 80%',
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Statistic='Average',
    Period=300,
    Threshold=80,
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    AlarmActions=['arn:aws:sns:us-east-1:123456789012:alerts']
)

# Get metric statistics
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=1)

response = cloudwatch.get_metric_statistics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    Dimensions=[
        {'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=3600,
    Statistics=['Average', 'Maximum']
)
for datapoint in response['Datapoints']:
    print(f"Time: {datapoint['Timestamp']}, Average: {datapoint['Average']}")`,
				},
				{
					Title: "CloudWatch Logs",
					Content: `CloudWatch Logs collects, monitors, and stores log files from applications and AWS services.

**CloudWatch Logs:**
- Centralized log storage
- Real-time monitoring
- Log retention (1 day to never)
- Log groups and streams
- Metric filters
- Log Insights queries

**Log Groups:**
- Container for log streams
- Retention policy
- Metric filters
- Subscription filters

**Log Streams:**
- Sequence of log events
- From same source
- Automatic or manual creation

**Log Insights:**
- Query log data
- SQL-like syntax
- Real-time analysis
- Save queries

**Best Practices:**
- Use structured logging
- Set appropriate retention
- Use metric filters for important events
- Monitor log ingestion
- Use Log Insights for analysis`,
					CodeExamples: `# Create log group
aws logs create-log-group --log-group-name /aws/lambda/my-function

# Put log events
aws logs put-log-events \\
    --log-group-name /aws/lambda/my-function \\
    --log-stream-name stream1 \\
    --log-events '[{"timestamp": 1640995200000, "message": "Application started"}]'

# Query logs with Log Insights
aws logs start-query \\
    --log-group-name /aws/lambda/my-function \\
    --start-time 1640995200 \\
    --end-time 1641081600 \\
    --query-string "fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc"

# Using boto3
import boto3
import time

logs = boto3.client('logs')

# Create log group
logs.create_log_group(logGroupName='/aws/lambda/my-function')

# Put log events
logs.create_log_stream(
    logGroupName='/aws/lambda/my-function',
    logStreamName='stream1'
)

logs.put_log_events(
    logGroupName='/aws/lambda/my-function',
    logStreamName='stream1',
    logEvents=[
        {
            'timestamp': int(time.time() * 1000),
            'message': 'Application started'
        }
    ]
)

# Query with Log Insights
response = logs.start_query(
    logGroupName='/aws/lambda/my-function',
    startTime=int(time.time() - 3600),
    endTime=int(time.time()),
    queryString='fields @timestamp, @message | filter @message like /ERROR/'
)

query_id = response['queryId']

# Wait for query to complete
import time
while True:
    result = logs.get_query_results(queryId=query_id)
    if result['status'] == 'Complete':
        break
    time.sleep(1)

for result in result['results']:
    print(result)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          307,
			Title:       "Amazon Route 53",
			Description: "Learn Route 53: DNS service, hosted zones, record types, health checks, and routing policies.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Route 53 Fundamentals",
					Content: `Amazon Route 53 is a scalable DNS web service for domain registration and DNS routing.

**Route 53 Features:**
- Domain registration
- DNS service
- Health checks
- Traffic routing policies
- Integration with AWS services

**Record Types:**
- **A**: IPv4 address
- **AAAA**: IPv6 address
- **CNAME**: Canonical name
- **MX**: Mail exchange
- **TXT**: Text records
- **NS**: Name servers
- **SOA**: Start of authority

**Hosted Zones:**
- **Public Hosted Zone**: Internet-facing DNS
- **Private Hosted Zone**: VPC-internal DNS

**Routing Policies:**
- **Simple**: Standard DNS routing
- **Weighted**: Distribute traffic by weight
- **Latency**: Route to lowest latency
- **Failover**: Active-passive failover
- **Geolocation**: Route by user location
- **Geoproximity**: Route by geographic location
- **Multivalue**: Multiple healthy records

**Best Practices:**
- Use appropriate routing policy
- Set up health checks
- Use alias records for AWS resources
- Monitor DNS queries
- Document DNS configuration`,
					CodeExamples: `# Create hosted zone
aws route53 create-hosted-zone \\
    --name example.com \\
    --caller-reference $(date +%s)

# Create A record
aws route53 change-resource-record-sets \\
    --hosted-zone-id Z1234567890ABC \\
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "www.example.com",
                "Type": "A",
                "TTL": 300,
                "ResourceRecords": [{"Value": "192.0.2.1"}]
            }
        }]
    }'

# Create alias record (ALB)
aws route53 change-resource-record-sets \\
    --hosted-zone-id Z1234567890ABC \\
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "api.example.com",
                "Type": "A",
                "AliasTarget": {
                    "HostedZoneId": "Z35SXDOTRQ7X7K",
                    "DNSName": "my-alb-1234567890.us-east-1.elb.amazonaws.com",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'

# Create weighted record
aws route53 change-resource-record-sets \\
    --hosted-zone-id Z1234567890ABC \\
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "app.example.com",
                "Type": "A",
                "SetIdentifier": "us-east-1",
                "Weight": 70,
                "TTL": 300,
                "ResourceRecords": [{"Value": "192.0.2.1"}]
            }
        }]
    }'

# Using boto3
import boto3

route53 = boto3.client('route53')

# Create hosted zone
response = route53.create_hosted_zone(
    Name='example.com',
    CallerReference=str(int(time.time()))
)
hosted_zone_id = response['HostedZone']['Id']

# Create A record
route53.change_resource_record_sets(
    HostedZoneId=hosted_zone_id,
    ChangeBatch={
        'Changes': [{
            'Action': 'CREATE',
            'ResourceRecordSet': {
                'Name': 'www.example.com',
                'Type': 'A',
                'TTL': 300,
                'ResourceRecords': [{'Value': '192.0.2.1'}]
            }
        }]
    }
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          308,
			Title:       "Amazon CloudFront",
			Description: "Learn CloudFront: CDN, distributions, origins, caching, and content delivery optimization.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "CloudFront Fundamentals",
					Content: `Amazon CloudFront is a content delivery network (CDN) that delivers content with low latency.

**CloudFront Features:**
- Global edge locations
- Caching at edge
- DDoS protection
- SSL/TLS termination
- Custom domain support
- Integration with AWS services

**Distribution Types:**
- **Web Distribution**: HTTP/HTTPS content
- **RTMP Distribution**: Streaming media (deprecated)

**Origins:**
- **S3 Bucket**: Static content
- **Custom Origin**: HTTP server
- **ALB/EC2**: Application load balancer
- **MediaStore**: Media storage

**Cache Behaviors:**
- Path patterns
- Origin selection
- Cache policies
- Origin request policies
- Viewer protocol policies

**Best Practices:**
- Use appropriate TTL values
- Invalidate cache when needed
- Use signed URLs for private content
- Enable compression
- Monitor cache hit ratio`,
					CodeExamples: `# Create CloudFront distribution
aws cloudfront create-distribution \\
    --distribution-config '{
        "CallerReference": "my-distribution-2024",
        "Comment": "My CloudFront distribution",
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3-my-bucket",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
            "ForwardedValues": {"QueryString": false},
            "MinTTL": 0,
            "DefaultTTL": 86400,
            "MaxTTL": 31536000
        },
        "Origins": {
            "Quantity": 1,
            "Items": [{
                "Id": "S3-my-bucket",
                "DomainName": "my-bucket.s3.amazonaws.com",
                "S3OriginConfig": {"OriginAccessIdentity": ""}
            }]
        },
        "Enabled": true
    }'

# Invalidate cache
aws cloudfront create-invalidation \\
    --distribution-id E1234567890ABC \\
    --paths "/*"

# Using boto3
import boto3

cloudfront = boto3.client('cloudfront')

# Create distribution
distribution = cloudfront.create_distribution(
    DistributionConfig={
        'CallerReference': 'my-distribution-2024',
        'Comment': 'My CloudFront distribution',
        'DefaultCacheBehavior': {
            'TargetOriginId': 'S3-my-bucket',
            'ViewerProtocolPolicy': 'redirect-to-https',
            'AllowedMethods': {'Quantity': 2, 'Items': ['GET', 'HEAD']},
            'ForwardedValues': {'QueryString': False},
            'MinTTL': 0,
            'DefaultTTL': 86400,
            'MaxTTL': 31536000
        },
        'Origins': {
            'Quantity': 1,
            'Items': [{
                'Id': 'S3-my-bucket',
                'DomainName': 'my-bucket.s3.amazonaws.com',
                'S3OriginConfig': {'OriginAccessIdentity': ''}
            }]
        },
        'Enabled': True
    }
)
print(f"Distribution ID: {distribution['Distribution']['Id']}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          309,
			Title:       "AWS CLI and SDKs",
			Description: "Learn AWS CLI and SDKs: command-line interface, SDKs for various languages, and automation.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "AWS CLI Fundamentals",
					Content: `AWS CLI is a unified tool to manage AWS services from the command line.

**AWS CLI Features:**
- Command-line interface
- Supports all AWS services
- Scriptable and automatable
- Cross-platform
- Configurable profiles

**Configuration:**
- Credentials (access key, secret key)
- Region
- Output format (json, yaml, text, table)
- Profiles for multiple accounts

**Common Commands:**
- aws configure: Set up credentials
- aws s3 ls: List S3 buckets
- aws ec2 describe-instances: List EC2 instances
- aws lambda invoke: Invoke Lambda function

**Best Practices:**
- Use IAM roles instead of access keys
- Use profiles for multiple accounts
- Use output filters for large responses
- Use pagination for large lists
- Secure credentials properly`,
					CodeExamples: `# Configure AWS CLI
aws configure
# Enter access key, secret key, region, output format

# Configure profile
aws configure --profile production

# Use profile
aws s3 ls --profile production

# List S3 buckets
aws s3 ls

# Copy file to S3
aws s3 cp file.txt s3://my-bucket/

# List EC2 instances
aws ec2 describe-instances

# Filter output
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId,State.Name]"

# Using boto3 (Python SDK)
import boto3

# Create clients
s3 = boto3.client('s3')
ec2 = boto3.client('ec2')

# List buckets
buckets = s3.list_buckets()
for bucket in buckets['Buckets']:
    print(bucket['Name'])

# List instances
instances = ec2.describe_instances()
for reservation in instances['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance: {instance['InstanceId']}, State: {instance['State']['Name']}")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
