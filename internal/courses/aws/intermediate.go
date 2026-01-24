package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          310,
			Title:       "Amazon RDS",
			Description: "Learn RDS: managed relational databases, multi-AZ deployments, read replicas, and backups.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "RDS vs DynamoDB (SQL vs NoSQL)",
					Content: `Choosing the right database is the most critical architectural decision.

**1. Amazon RDS (Relational Database Service)**
*   **What is it:** Managed SQL (MySQL, Postgres, Aurora).
*   **Use case:** Complex relationships, transactions (ACID), e-commerce systems, legacy app migration.
*   **Scaling:** Vertical (bigger instance). Harder to scale horizontally (read replicas help, but write is single-master except for Multi-Master Aurora).

**2. Amazon DynamoDB (NoSQL)**
*   **What is it:** Serverless key-value store.
*   **Use case:** Massive scale, simple lookup patterns (shopping cart, user session, gaming leaderboard).
*   **Scaling:** Horizontal (more partitions). Can handle millions of requests/sec. Single-digit millisecond latency.

**Decision Matrix:**
*   Need joins like ` + "`SELECT * FROM Users JOIN Orders`" + `? -> **RDS**
*   Need consistent schema? -> **RDS**
*   Need to scale to 10M users with zero maintenance? -> **DynamoDB**
*   storing JSON documents? -> **DynamoDB** (or DocumentDB)

**Modern Pattern:**
Use **DynamoDB** as the default for new microservices. Use **Aurora** if you strictly need relational features.`,
					CodeExamples: `# Create RDS (SQL)
aws rds create-db-instance --db-instance-identifier mydb --engine mysql --db-instance-class db.t3.micro

# Create DynamoDB (NoSQL)
aws dynamodb create-table \\
    --table-name Users \\
    --attribute-definitions AttributeName=UserId,AttributeType=S \\
    --key-schema AttributeName=UserId,KeyType=HASH \\
    --billing-mode PAY_PER_REQUEST`,
				},
				{
					Title: "RDS Backups and Snapshots",
					Content: `RDS provides automated backups and manual snapshots for data protection and recovery.

**Automated Backups:**
- Enabled by default (retention: 1-35 days)
- Daily full backup during backup window
- Transaction logs backed up every 5 minutes
- Point-in-time recovery (PITR) supported
- Stored in S3 (no additional charge)

**Manual Snapshots:**
- User-initiated backups
- Retained until explicitly deleted
- Can copy across regions
- Can share with other accounts
- Useful for long-term retention

**Backup Window:**
- Configurable time window
- Low I/O impact preferred
- UTC timezone
- 30-minute window
- Can't overlap with maintenance window

**Restore Operations:**
- Restore to point in time
- Restore from snapshot
- Creates new DB instance
- Can restore to different instance class
- Can restore to different AZ

**Backup Best Practices:**
- Enable automated backups
- Set appropriate retention period
- Test restore procedures
- Copy snapshots to other regions
- Monitor backup status

**Best Practices:**
- Use automated backups for production
- Create manual snapshots before major changes
- Test restore procedures regularly
- Copy snapshots for disaster recovery
- Monitor backup storage usage`,
					CodeExamples: `# Modify backup retention
aws rds modify-db-instance \\
    --db-instance-identifier mydb \\
    --backup-retention-period 7 \\
    --preferred-backup-window "03:00-04:00"

# Create manual snapshot
aws rds create-db-snapshot \\
    --db-instance-identifier mydb \\
    --db-snapshot-identifier mydb-snapshot-2024-01-01

# Copy snapshot to another region
aws rds copy-db-snapshot \\
    --source-db-snapshot-identifier mydb-snapshot-2024-01-01 \\
    --target-db-snapshot-identifier mydb-snapshot-2024-01-01-copy \\
    --source-region us-east-1 \\
    --region us-west-2

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \\
    --db-instance-identifier mydb-restored \\
    --db-snapshot-identifier mydb-snapshot-2024-01-01 \\
    --db-instance-class db.t3.small

# Restore to point in time
aws rds restore-db-instance-to-point-in-time \\
    --source-db-instance-identifier mydb \\
    --target-db-instance-identifier mydb-restored-pit \\
    --restore-time 2024-01-01T12:00:00Z

# List snapshots
aws rds describe-db-snapshots \\
    --db-instance-identifier mydb

# Delete snapshot
aws rds delete-db-snapshot \\
    --db-snapshot-identifier mydb-snapshot-2024-01-01

# Share snapshot with another account
aws rds modify-db-snapshot-attribute \\
    --db-snapshot-identifier mydb-snapshot-2024-01-01 \\
    --attribute-name restore \\
    --values-to-add 123456789012

# Using boto3
import boto3
from datetime import datetime, timedelta

rds = boto3.client('rds')

# Modify backup retention
rds.modify_db_instance(
    DBInstanceIdentifier='mydb',
    BackupRetentionPeriod=7,
    PreferredBackupWindow='03:00-04:00',
    ApplyImmediately=True
)

# Create manual snapshot
snapshot = rds.create_db_snapshot(
    DBInstanceIdentifier='mydb',
    DBSnapshotIdentifier=f'mydb-snapshot-{datetime.now().strftime("%Y%m%d")}'
)
print(f"Snapshot ID: {snapshot['DBSnapshot']['DBSnapshotIdentifier']}")

# Copy snapshot to another region
rds_west = boto3.client('rds', region_name='us-west-2')
rds_west.copy_db_snapshot(
    SourceDBSnapshotIdentifier='arn:aws:rds:us-east-1:123456789012:snapshot:mydb-snapshot-20240101',
    TargetDBSnapshotIdentifier='mydb-snapshot-20240101-copy',
    SourceRegion='us-east-1'
)

# Restore from snapshot
restored = rds.restore_db_instance_from_db_snapshot(
    DBInstanceIdentifier='mydb-restored',
    DBSnapshotIdentifier='mydb-snapshot-2024-01-01',
    DBInstanceClass='db.t3.small'
)

# Restore to point in time
pit_restore = rds.restore_db_instance_to_point_in_time(
    SourceDBInstanceIdentifier='mydb',
    TargetDBInstanceIdentifier='mydb-restored-pit',
    RestoreTime=datetime(2024, 1, 1, 12, 0, 0),
    DBInstanceClass='db.t3.small'
)

# List snapshots
snapshots = rds.describe_db_snapshots(
    DBInstanceIdentifier='mydb'
)
for snapshot in snapshots['DBSnapshots']:
    print(f"Snapshot: {snapshot['DBSnapshotIdentifier']}, Status: {snapshot['Status']}")`,
				},
				{
					Title: "RDS Performance Insights",
					Content: `Performance Insights provides database performance monitoring and analysis for RDS instances.

**Performance Insights:**
- Real-time database performance monitoring
- Identifies database bottlenecks
- Shows top SQL statements
- Visualizes wait events
- Retention: 7 days (free) or up to 2 years (paid)

**Key Metrics:**
- Database load (DB Load)
- Average active sessions (AAS)
- Wait events
- Top SQL statements
- Dimensions (database, host, user)

**Wait Events:**
- CPU: High CPU usage
- IO: Disk I/O bottlenecks
- Lock: Lock contention
- Network: Network latency
- Other: Other wait types

**Use Cases:**
- Identify slow queries
- Diagnose performance issues
- Optimize database performance
- Capacity planning
- Troubleshooting

**Best Practices:**
- Enable for production databases
- Monitor regularly
- Analyze wait events
- Optimize top SQL statements
- Use extended retention for critical DBs

**Integration:**
- CloudWatch integration
- Can export to CloudWatch Logs
- API access for automation
- Dashboard visualization`,
					CodeExamples: `# Enable Performance Insights
aws rds modify-db-instance \\
    --db-instance-identifier mydb \\
    --enable-performance-insights \\
    --performance-insights-retention-period 7

# Enable with extended retention (paid)
aws rds modify-db-instance \\
    --db-instance-identifier mydb \\
    --enable-performance-insights \\
    --performance-insights-retention-period 731 \\
    --performance-insights-kms-key-id alias/my-kms-key

# Disable Performance Insights
aws rds modify-db-instance \\
    --db-instance-identifier mydb \\
    --enable-performance-insights false

# Get Performance Insights metrics (using boto3)
import boto3
from datetime import datetime, timedelta

rds = boto3.client('rds')
pi = boto3.client('pi')

# Get DB instance ARN
db_instance = rds.describe_db_instances(
    DBInstanceIdentifier='mydb'
)['DBInstances'][0]
db_arn = db_instance['DBInstanceArn']

# Get performance insights data
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

response = pi.get_resource_metrics(
    ServiceType='RDS',
    Identifier=db_arn,
    MetricQueries=[
        {
            'Metric': 'db.load.avg',
            'GroupBy': {
                'Group': 'db.sql_tokenized.statement'
            }
        }
    ],
    StartTime=start_time,
    EndTime=end_time,
    PeriodInSeconds=60
)

# Get dimension values (top SQL statements)
dimensions = pi.get_dimension_values(
    ServiceType='RDS',
    Identifier=db_arn,
    Metric='db.load.avg',
    GroupBy='db.sql_tokenized.statement',
    StartTime=start_time,
    EndTime=end_time
)

for dimension in dimensions['DimensionValues']:
    print(f"SQL: {dimension['Value']}, Total: {dimension['Total']}")

# Get wait events
wait_events = pi.get_dimension_values(
    ServiceType='RDS',
    Identifier=db_arn,
    Metric='db.load.avg',
    GroupBy='wait_event.name',
    StartTime=start_time,
    EndTime=end_time
)

for event in wait_events['DimensionValues']:
    print(f"Wait Event: {event['Value']}, Total: {event['Total']}")

# Using CloudWatch for Performance Insights
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get Performance Insights metrics via CloudWatch
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/RDS',
    MetricName='PerformanceInsightsDBLoad',
    Dimensions=[
        {'Name': 'DBInstanceIdentifier', 'Value': 'mydb'}
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=60,
    Statistics=['Average']
)

for datapoint in response['Datapoints']:
    print(f"Time: {datapoint['Timestamp']}, DB Load: {datapoint['Average']}")`,
				},
				{
					Title: "RDS Parameter Groups",
					Content: `Parameter Groups allow you to configure database engine parameters for RDS instances.

**Parameter Groups:**
- Container for engine configuration
- Default parameter groups provided
- Custom parameter groups for tuning
- Applied at instance level
- Some changes require reboot

**Parameter Types:**
- **Static**: Require instance reboot
- **Dynamic**: Applied immediately
- **Engine-specific**: Vary by database engine

**Use Cases:**
- Tune database performance
- Configure connection limits
- Set query timeout
- Configure logging
- Enable/disable features

**Best Practices:**
- Create custom parameter groups
- Test changes in non-production first
- Document parameter changes
- Use descriptive names
- Keep parameter groups organized

**Parameter Modification:**
- Modify parameter group
- Apply to DB instance
- Reboot if required (static parameters)
- Monitor impact of changes`,
					CodeExamples: `# Create DB parameter group
aws rds create-db-parameter-group \\
    --db-parameter-group-name my-mysql-params \\
    --db-parameter-group-family mysql8.0 \\
    --description "Custom MySQL parameters"

# Modify parameter
aws rds modify-db-parameter-group \\
    --db-parameter-group-name my-mysql-params \\
    --parameters "ParameterName=max_connections,ParameterValue=200,ApplyMethod=immediate"

# Modify static parameter (requires reboot)
aws rds modify-db-parameter-group \\
    --db-parameter-group-name my-mysql-params \\
    --parameters "ParameterName=innodb_buffer_pool_size,ParameterValue=1073741824,ApplyMethod=pending-reboot"

# Apply parameter group to instance
aws rds modify-db-instance \\
    --db-instance-identifier mydb \\
    --db-parameter-group-name my-mysql-params \\
    --apply-immediately

# Reboot instance to apply static parameters
aws rds reboot-db-instance \\
    --db-instance-identifier mydb

# Describe parameter group
aws rds describe-db-parameters \\
    --db-parameter-group-name my-mysql-params

# Reset parameter to default
aws rds reset-db-parameter-group \\
    --db-parameter-group-name my-mysql-params \\
    --parameters "ParameterName=max_connections,ResetAllParameters=false"

# Using boto3
import boto3

rds = boto3.client('rds')

# Create parameter group
pg = rds.create_db_parameter_group(
    DBParameterGroupName='my-mysql-params',
    DBParameterGroupFamily='mysql8.0',
    Description='Custom MySQL parameters'
)
print(f"Parameter Group: {pg['DBParameterGroup']['DBParameterGroupName']}")

# Modify parameters
rds.modify_db_parameter_group(
    DBParameterGroupName='my-mysql-params',
    Parameters=[
        {
            'ParameterName': 'max_connections',
            'ParameterValue': '200',
            'ApplyMethod': 'immediate'
        },
        {
            'ParameterName': 'innodb_buffer_pool_size',
            'ParameterValue': '1073741824',
            'ApplyMethod': 'pending-reboot'
        }
    ]
)

# Apply to instance
rds.modify_db_instance(
    DBInstanceIdentifier='mydb',
    DBParameterGroupName='my-mysql-params',
    ApplyImmediately=True
)

# Get parameter values
parameters = rds.describe_db_parameters(
    DBParameterGroupName='my-mysql-params'
)
for param in parameters['Parameters']:
    if param['Source'] == 'user':
        print(f"{param['ParameterName']}: {param['ParameterValue']}")

# Reboot to apply static parameters
rds.reboot_db_instance(
    DBInstanceIdentifier='mydb'
)

# Wait for reboot to complete
waiter = rds.get_waiter('db_instance_available')
waiter.wait(DBInstanceIdentifier='mydb')`,
				},
				{
					Title: "Amazon Aurora",
					Content: `Amazon Aurora is a MySQL and PostgreSQL-compatible relational database with high performance and availability.

**Aurora Features:**
- MySQL and PostgreSQL compatible
- Up to 5x faster than MySQL
- Up to 3x faster than PostgreSQL
- Automatic storage scaling (10GB to 128TB)
- 6 copies across 3 AZs
- Continuous backup to S3

**Aurora Architecture:**
- Storage layer separate from compute
- Shared storage volume
- Automatic failover (< 30 seconds)
- Read replicas share storage
- Up to 15 read replicas

**Aurora Serverless:**
- Automatic scaling
- Pay per use
- Pause when inactive
- Good for variable workloads
- Compatible with Aurora

**Aurora Global Database:**
- Cross-region replication
- < 1 second replication lag
- Disaster recovery
- Global read scaling
- Up to 5 secondary regions

**Backup and Restore:**
- Continuous backup (point-in-time recovery)
- Backup retention: 1-35 days
- Manual snapshots
- Fast clone (copy-on-write)
- Cross-region snapshot copy

**Best Practices:**
- Use Aurora for high-performance needs
- Use Aurora Serverless for variable workloads
- Enable Multi-AZ for production
- Use read replicas for read scaling
- Monitor performance metrics`,
					CodeExamples: `# Create Aurora cluster
aws rds create-db-cluster \\
    --db-cluster-identifier my-aurora-cluster \\
    --engine aurora-mysql \\
    --engine-version 5.7.mysql_aurora.2.10.0 \\
    --master-username admin \\
    --master-user-password MyPassword123 \\
    --database-name mydb

# Create Aurora instance
aws rds create-db-instance \\
    --db-instance-identifier my-aurora-instance-1 \\
    --db-instance-class db.r5.large \\
    --engine aurora-mysql \\
    --db-cluster-identifier my-aurora-cluster

# Create Aurora read replica
aws rds create-db-instance \\
    --db-instance-identifier my-aurora-replica-1 \\
    --db-instance-class db.r5.large \\
    --engine aurora-mysql \\
    --db-cluster-identifier my-aurora-cluster

# Create Aurora Serverless cluster
aws rds create-db-cluster \\
    --db-cluster-identifier my-aurora-serverless \\
    --engine aurora-mysql \\
    --engine-mode serverless \\
    --scaling-configuration MinCapacity=2,MaxCapacity=16 \\
    --master-username admin \\
    --master-user-password MyPassword123

# Create Aurora Global Database
# Create primary cluster
aws rds create-db-cluster \\
    --db-cluster-identifier my-aurora-global-primary \\
    --engine aurora-mysql \\
    --master-username admin \\
    --master-user-password MyPassword123

# Create global cluster
aws rds create-global-cluster \\
    --global-cluster-identifier my-global-cluster \\
    --source-db-cluster-identifier my-aurora-global-primary

# Add secondary region
aws rds create-db-cluster \\
    --db-cluster-identifier my-aurora-global-secondary \\
    --engine aurora-mysql \\
    --global-cluster-identifier my-global-cluster

# Fast clone (copy-on-write)
aws rds restore-db-cluster-from-snapshot \\
    --db-cluster-identifier my-aurora-clone \\
    --snapshot-identifier my-aurora-snapshot \\
    --engine aurora-mysql

# Using boto3
import boto3

rds = boto3.client('rds')

# Create Aurora cluster
cluster = rds.create_db_cluster(
    DBClusterIdentifier='my-aurora-cluster',
    Engine='aurora-mysql',
    EngineVersion='5.7.mysql_aurora.2.10.0',
    MasterUsername='admin',
    MasterUserPassword='MyPassword123',
    DatabaseName='mydb'
)

# Create primary instance
primary = rds.create_db_instance(
    DBInstanceIdentifier='my-aurora-instance-1',
    DBInstanceClass='db.r5.large',
    Engine='aurora-mysql',
    DBClusterIdentifier='my-aurora-cluster'
)

# Create read replica
replica = rds.create_db_instance(
    DBInstanceIdentifier='my-aurora-replica-1',
    DBInstanceClass='db.r5.large',
    Engine='aurora-mysql',
    DBClusterIdentifier='my-aurora-cluster'
)

# Create Aurora Serverless
serverless = rds.create_db_cluster(
    DBClusterIdentifier='my-aurora-serverless',
    Engine='aurora-mysql',
    EngineMode='serverless',
    ScalingConfiguration={
        'MinCapacity': 2,
        'MaxCapacity': 16
    },
    MasterUsername='admin',
    MasterUserPassword='MyPassword123'
)

# Get cluster endpoints
cluster_info = rds.describe_db_clusters(
    DBClusterIdentifier='my-aurora-cluster'
)['DBClusters'][0]

print(f"Writer Endpoint: {cluster_info['Endpoint']}")
print(f"Reader Endpoint: {cluster_info['ReaderEndpoint']}")

# List cluster members
members = rds.describe_db_clusters(
    DBClusterIdentifier='my-aurora-cluster'
)['DBClusters'][0]['DBClusterMembers']

for member in members:
    print(f"Instance: {member['DBInstanceIdentifier']}, Role: {member['IsClusterWriter']}")`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          311,
			Title:       "AWS Lambda",
			Description: "Master serverless computing with Lambda: functions, triggers, event sources, and best practices.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "Serverless Patterns: Lambda & API Gateway",
					Content: `The most common serverless pattern in AWS is the **API Gateway -> Lambda -> DynamoDB** stack.

**Architecture Flow:**
1.  **Client** sends HTTP Request (GET /users).
2.  **API Gateway** receives it, checks authorization (Cognito/IAM), and triggers Lambda.
3.  **Lambda** (Compute) spins up, runs your code (logic), talks to the DB.
4.  **DynamoDB** (Storage) returns the data.
5.  **Lambda** returns JSON to API Gateway -> Client.

**Why this rocks:**
*   **Scale to Zero:** Cost is $0.00 if no one visits your site.
*   **Infinite Scale:** Can handle 10,000 requests/second instantly.
*   **No Ops:** No OS to patch, no servers to reboot.

**Key best practices:**
*   **Stateless:** Lambda functions are ephemeral. Never save files to local disk expecting them to be there next time. Use S3.
*   **Cold Starts:** Initial request might take 1-2s. Keep functions small or use "Provisioned Concurrency" for critical paths.
*   **Timeouts:** API Gateway has a hard 29s timeout. For long tasks, use SQS (Queue) in between.`,
					CodeExamples: `# 1. Create a Lambda Function
aws lambda create-function \\
    --function-name GetUser \\
    --runtime python3.9 \\
    --role arn:aws:iam::123456789012:role/service-role/role \\
    --handler lambda_function.lambda_handler \\
    --zip-file fileb://function.zip

# 2. Create an API Gateway (HTTP API is cheaper/faster than REST API)
aws apigatewayv2 create-api \\
    --name my-api \\
    --protocol-type HTTP \\
    --target arn:aws:lambda:us-east-1:123456789012:function:GetUser

# 3. Invoke it directly to test
aws lambda invoke --function-name GetUser response.json`,
				},
				{
					Title: "Lambda Triggers and Event Sources",
					Content: `Lambda functions can be triggered by various AWS services and events.

**Common Event Sources:**
- **API Gateway**: HTTP/HTTPS requests
- **S3**: Object created, deleted, or modified
- **DynamoDB**: Streams (inserts, updates, deletes)
- **SNS**: Notifications
- **SQS**: Message queue
- **CloudWatch Events**: Scheduled or event-based
- **Kinesis**: Data streams
- **CodeCommit**: Git events
- **CloudFormation**: Stack events

**Event Patterns:**
- S3: Object key, bucket name
- DynamoDB: Stream records
- CloudWatch Events: JSON event pattern
- API Gateway: HTTP request/response

**Asynchronous Invocations:**
- S3, SNS, CloudWatch Events
- Lambda retries on failure
- Dead letter queue (DLQ) for failed invocations
- Configure retry attempts

**Synchronous Invocations:**
- API Gateway, CloudFront
- Wait for response
- Error returned immediately
- No automatic retries

**Best Practices:**
- Use appropriate event source
- Handle errors gracefully
- Set up DLQ for async invocations
- Monitor function metrics
- Use appropriate timeout`,
					CodeExamples: `# Create S3 trigger
aws lambda add-permission \\
    --function-name process-s3-object \\
    --statement-id s3-trigger \\
    --action lambda:InvokeFunction \\
    --principal s3.amazonaws.com \\
    --source-arn arn:aws:s3:::my-bucket

# Configure S3 bucket notification
aws s3api put-bucket-notification-configuration \\
    --bucket my-bucket \\
    --notification-configuration file://notification.json

# notification.json
{
    "LambdaFunctionConfigurations": [
        {
            "LambdaFunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:process-s3-object",
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

# Create CloudWatch Events rule
aws events put-rule \\
    --name daily-backup \\
    --schedule-expression "cron(0 2 * * ? *)"

# Add Lambda as target
aws events put-targets \\
    --rule daily-backup \\
    --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:123456789012:function:backup-function"

# Create API Gateway trigger (via console or CloudFormation)
# API Gateway automatically integrates with Lambda

# Configure dead letter queue
aws lambda update-function-configuration \\
    --function-name my-function \\
    --dead-letter-config TargetArn=arn:aws:sqs:us-east-1:123456789012:dlq

# Using boto3
import boto3

lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')
events_client = boto3.client('events')

# Add S3 permission
lambda_client.add_permission(
    FunctionName='process-s3-object',
    StatementId='s3-trigger',
    Action='lambda:InvokeFunction',
    Principal='s3.amazonaws.com',
    SourceArn='arn:aws:s3:::my-bucket'
)

# Configure S3 notification
s3_client.put_bucket_notification_configuration(
    Bucket='my-bucket',
    NotificationConfiguration={
        'LambdaFunctionConfigurations': [
            {
                'LambdaFunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:process-s3-object',
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {
                                'Name': 'prefix',
                                'Value': 'uploads/'
                            }
                        ]
                    }
                }
            }
        ]
    }
)

# Create scheduled rule
events_client.put_rule(
    Name='daily-backup',
    ScheduleExpression='cron(0 2 * * ? *)'
)

events_client.put_targets(
    Rule='daily-backup',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:backup-function'
        }
    ]
)`,
				},
				{
					Title: "Lambda Layers",
					Content: `Lambda Layers allow you to package libraries, custom runtimes, and other function dependencies separately.

**Lambda Layers:**
- Share code and data across functions
- Reduce deployment package size
- Version and manage dependencies separately
- Up to 5 layers per function
- Layers extracted to /opt directory

**Layer Contents:**
- Libraries and dependencies
- Custom runtimes
- Configuration files
- Data files
- Binaries

**Layer Structure:**
- Must follow specific directory structure
- Runtime-specific paths (python/, nodejs/, etc.)
- Common paths for all runtimes (/opt)

**Layer Versions:**
- Each layer has versions
- Can reference specific version or $LATEST
- Old versions remain available
- Can't delete versions in use

**Use Cases:**
- Share common libraries
- Reduce deployment package size
- Manage dependencies centrally
- Custom runtimes
- Common utilities

**Best Practices:**
- Use layers for large dependencies
- Keep layers focused and small
- Version layers properly
- Test layer compatibility
- Monitor layer size limits`,
					CodeExamples: `# Create layer structure
mkdir -p layer/python/lib/python3.11/site-packages
pip install requests -t layer/python/lib/python3.11/site-packages/

# Create layer ZIP
cd layer
zip -r ../my-layer.zip .
cd ..

# Publish layer
aws lambda publish-layer-version \\
    --layer-name my-dependencies \\
    --description "Common Python dependencies" \\
    --zip-file fileb://my-layer.zip \\
    --compatible-runtimes python3.11 python3.10

# Create function with layer
aws lambda create-function \\
    --function-name my-function \\
    --runtime python3.11 \\
    --role arn:aws:iam::123456789012:role/lambda-execution-role \\
    --handler lambda_function.lambda_handler \\
    --zip-file fileb://function.zip \\
    --layers arn:aws:lambda:us-east-1:123456789012:layer:my-dependencies:1

# Add layer to existing function
aws lambda update-function-configuration \\
    --function-name my-function \\
    --layers arn:aws:lambda:us-east-1:123456789012:layer:my-dependencies:1

# List layer versions
aws lambda list-layer-versions \\
    --layer-name my-dependencies

# Get layer version
aws lambda get-layer-version \\
    --layer-name my-dependencies \\
    --version-number 1

# Delete layer version
aws lambda delete-layer-version \\
    --layer-name my-dependencies \\
    --version-number 1

# Using boto3
import boto3
import zipfile
import os

lambda_client = boto3.client('lambda')

# Create layer ZIP
layer_dir = 'layer/python/lib/python3.11/site-packages'
os.makedirs(layer_dir, exist_ok=True)
os.system(f'pip install requests -t {layer_dir}')

with zipfile.ZipFile('my-layer.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('layer'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'layer')
            zipf.write(file_path, arcname)

# Publish layer
with open('my-layer.zip', 'rb') as layer_zip:
    response = lambda_client.publish_layer_version(
        LayerName='my-dependencies',
        Description='Common Python dependencies',
        Content={'ZipFile': layer_zip.read()},
        CompatibleRuntimes=['python3.11', 'python3.10']
    )
layer_arn = response['LayerVersionArn']
print(f"Layer ARN: {layer_arn}")

# Update function with layer
lambda_client.update_function_configuration(
    FunctionName='my-function',
    Layers=[layer_arn]
)

# List layers for function
function_config = lambda_client.get_function_configuration(
    FunctionName='my-function'
)
print(f"Layers: {function_config.get('Layers', [])}")`,
				},
				{
					Title: "Lambda Performance Tuning",
					Content: `Optimizing Lambda function performance involves configuring memory, timeout, and code optimization.

**Memory Configuration:**
- Memory allocation (128MB to 10GB)
- CPU power scales with memory
- More memory = more CPU
- Cost increases with memory
- Find optimal memory for your workload

**Timeout Configuration:**
- Maximum execution time (1 second to 15 minutes)
- Set based on expected duration
- Include buffer for variability
- Monitor timeout errors

**Cold Starts:**
- First invocation initializes runtime
- Varies by runtime and package size
- Provisioned concurrency reduces cold starts
- Use layers to reduce package size
- Keep initialization code minimal

**Code Optimization:**
- Minimize package size
- Use layers for dependencies
- Initialize clients outside handler
- Reuse connections
- Optimize imports

**Concurrency:**
- Default: 1000 concurrent executions per region
- Can request limit increase
- Reserved concurrency limits function
- Provisioned concurrency pre-warms functions

**Best Practices:**
- Right-size memory allocation
- Minimize cold starts
- Optimize package size
- Use connection pooling
- Monitor performance metrics
- Use X-Ray for tracing`,
					CodeExamples: `# Update function memory
aws lambda update-function-configuration \\
    --function-name my-function \\
    --memory-size 512

# Update function timeout
aws lambda update-function-configuration \\
    --function-name my-function \\
    --timeout 30

# Configure reserved concurrency
aws lambda put-function-concurrency \\
    --function-name my-function \\
    --reserved-concurrent-executions 10

# Remove reserved concurrency
aws lambda delete-function-concurrency \\
    --function-name my-function

# Configure provisioned concurrency
aws lambda put-provisioned-concurrency-config \\
    --function-name my-function \\
    --qualifier \$LATEST \\
    --provisioned-concurrent-executions 5

# Optimized Lambda function (Python)
import json
import boto3
import os

# Initialize clients outside handler (reused across invocations)
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

# Reuse table reference
table = dynamodb.Table(os.environ['TABLE_NAME'])

def lambda_handler(event, context):
    # Handler code here
    # Clients and table are already initialized
    response = table.get_item(Key={'id': event['id']})
    return {
        'statusCode': 200,
        'body': json.dumps(response['Item'])
    }

# Using boto3
import boto3

lambda_client = boto3.client('lambda')

# Update memory and timeout
lambda_client.update_function_configuration(
    FunctionName='my-function',
    MemorySize=512,
    Timeout=30
)

# Set reserved concurrency
lambda_client.put_function_concurrency(
    FunctionName='my-function',
    ReservedConcurrentExecutions=10
)

# Configure provisioned concurrency
lambda_client.put_provisioned_concurrency_config(
    FunctionName='my-function',
    Qualifier='$LATEST',
    ProvisionedConcurrentExecutions=5
)

# Get function configuration
config = lambda_client.get_function_configuration(
    FunctionName='my-function'
)
print(f"Memory: {config['MemorySize']}MB")
print(f"Timeout: {config['Timeout']}s")
print(f"Reserved Concurrency: {config.get('ReservedConcurrentExecutions', 'Unlimited')}")

# Monitor performance with CloudWatch
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get duration metric
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/Lambda',
    MetricName='Duration',
    Dimensions=[
        {'Name': 'FunctionName', 'Value': 'my-function'}
    ],
    StartTime='2024-01-01T00:00:00Z',
    EndTime='2024-01-01T23:59:59Z',
    Period=3600,
    Statistics=['Average', 'Maximum']
)
print(response)`,
				},
				{
					Title: "Lambda Error Handling and Retries",
					Content: `Proper error handling and retry configuration is crucial for reliable Lambda functions.

**Error Types:**
- **Handled Errors**: Caught and processed in code
- **Unhandled Errors**: Exceptions not caught
- **Timeouts**: Function exceeds timeout
- **Out of Memory**: Function exceeds memory limit

**Retry Behavior:**
- **Synchronous Invocations**: No automatic retries
- **Asynchronous Invocations**: Automatic retries (up to 2)
- **Stream-based Invocations**: Retries until success or data expires

**Dead Letter Queues (DLQ):**
- SQS queue or SNS topic
- Receives failed invocations
- Configure per function
- Helps debug failures
- Prevents infinite retries

**Error Handling Strategies:**
- Try-catch blocks
- Return appropriate status codes
- Log errors for debugging
- Use DLQ for async invocations
- Implement idempotency

**Asynchronous Invocations:**
- S3, SNS, CloudWatch Events
- Automatic retries (2 attempts)
- Failed invocations go to DLQ
- Can configure retry attempts

**Best Practices:**
- Handle all expected errors
- Use DLQ for async invocations
- Implement idempotency
- Log errors appropriately
- Monitor error rates
- Set appropriate timeouts`,
					CodeExamples: `# Configure Dead Letter Queue
aws lambda update-function-configuration \\
    --function-name my-function \\
    --dead-letter-config TargetArn=arn:aws:sqs:us-east-1:123456789012:dlq

# Remove Dead Letter Queue
aws lambda update-function-configuration \\
    --function-name my-function \\
    --dead-letter-config '{}'

# Create DLQ
aws sqs create-queue \\
    --queue-name lambda-dlq \\
    --attributes '{"MessageRetentionPeriod": "1209600"}'

# Lambda function with error handling
import json
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Process event
        result = process_event(event)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # Re-raise for async invocations to trigger retry/DLQ
        raise

def process_event(event):
    # Validate input
    if 'data' not in event:
        raise ValueError("Missing 'data' field")
    
    # Process data
    return {'processed': True}

# Idempotent Lambda function
import json
import boto3
import hashlib

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('processed-events')

def lambda_handler(event, context):
    # Generate idempotency key
    event_id = event.get('id') or hashlib.md5(
        json.dumps(event, sort_keys=True).encode()
    ).hexdigest()
    
    # Check if already processed
    try:
        response = table.get_item(Key={'id': event_id})
        if 'Item' in response:
            # Already processed, return cached result
            return {
                'statusCode': 200,
                'body': json.dumps(response['Item']['result'])
            }
    except Exception as e:
        logger.error(f"Error checking idempotency: {str(e)}")
    
    # Process event
    try:
        result = process_event(event)
        
        # Store result
        table.put_item(
            Item={
                'id': event_id,
                'result': result,
                'timestamp': context.aws_request_id
            }
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        raise

# Using boto3
import boto3

lambda_client = boto3.client('lambda')
sqs = boto3.client('sqs')

# Create DLQ
dlq_response = sqs.create_queue(
    QueueName='lambda-dlq',
    Attributes={
        'MessageRetentionPeriod': '1209600'  # 14 days
    }
)
dlq_arn = dlq_response['QueueAttributes']['QueueArn']

# Configure DLQ
lambda_client.update_function_configuration(
    FunctionName='my-function',
    DeadLetterConfig={
        'TargetArn': dlq_arn
    }
)

# Get function configuration
config = lambda_client.get_function_configuration(
    FunctionName='my-function'
)
print(f"DLQ: {config.get('DeadLetterConfig', {}).get('TargetArn', 'Not configured')}")

# Process DLQ messages
dlq_url = sqs.get_queue_url(QueueName='lambda-dlq')['QueueUrl']
messages = sqs.receive_message(
    QueueUrl=dlq_url,
    MaxNumberOfMessages=10
)

for message in messages.get('Messages', []):
    print(f"Failed invocation: {message['Body']}")
    # Analyze and fix issue
    # Delete message after processing
    sqs.delete_message(
        QueueUrl=dlq_url,
        ReceiptHandle=message['ReceiptHandle']
    )`,
				},
				{
					Title: "Lambda VPC Integration",
					Content: `Lambda functions can access resources in VPCs, but require proper network configuration.

**VPC Configuration:**
- Lambda ENI created in VPC subnets
- Security groups control access
- Can access VPC resources (RDS, ElastiCache, etc.)
- Cannot access internet by default
- Requires NAT Gateway for internet access

**VPC Configuration Requirements:**
- Subnet IDs (private recommended)
- Security group IDs
- ENI created automatically
- Cold start may be longer

**Internet Access:**
- Default: No internet access from VPC
- Use NAT Gateway for outbound internet
- Use VPC Endpoints for AWS services
- Consider cost implications

**Security Groups:**
- Control inbound/outbound traffic
- Allow Lambda to access resources
- Allow resources to access Lambda (if needed)
- Use least privilege principle

**Use Cases:**
- Access RDS databases
- Access ElastiCache clusters
- Access private APIs
- Access resources in VPC

**Best Practices:**
- Use private subnets
- Configure security groups properly
- Use VPC Endpoints when possible
- Minimize VPC configuration when not needed
- Monitor ENI limits
- Consider cold start impact`,
					CodeExamples: `# Configure Lambda VPC access
aws lambda update-function-configuration \\
    --function-name my-function \\
    --vpc-config SubnetIds=subnet-12345678,subnet-87654321,SecurityGroupIds=sg-12345678

# Remove VPC configuration
aws lambda update-function-configuration \\
    --function-name my-function \\
    --vpc-config '{}'

# Lambda function accessing RDS in VPC
import json
import pymysql
import os

def lambda_handler(event, context):
    # RDS connection (in VPC)
    connection = pymysql.connect(
        host=os.environ['RDS_HOST'],
        user=os.environ['RDS_USER'],
        password=os.environ['RDS_PASSWORD'],
        database=os.environ['RDS_DB'],
        connect_timeout=5
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users LIMIT 10")
            results = cursor.fetchall()
        
        return {
            'statusCode': 200,
            'body': json.dumps([dict(row) for row in results])
        }
    finally:
        connection.close()

# Lambda function with VPC and internet access (via NAT)
import json
import requests

def lambda_handler(event, context):
    # Can access internet via NAT Gateway
    response = requests.get('https://api.example.com/data')
    return {
        'statusCode': 200,
        'body': json.dumps(response.json())
    }

# Using boto3
import boto3

lambda_client = boto3.client('lambda')

# Configure VPC
lambda_client.update_function_configuration(
    FunctionName='my-function',
    VpcConfig={
        'SubnetIds': ['subnet-12345678', 'subnet-87654321'],
        'SecurityGroupIds': ['sg-12345678']
    }
)

# Wait for VPC configuration to be ready
import time
waiter = lambda_client.get_waiter('function_updated')
waiter.wait(FunctionName='my-function')

# Get VPC configuration
config = lambda_client.get_function_configuration(
    FunctionName='my-function'
)
vpc_config = config.get('VpcConfig', {})
print(f"VPC ID: {vpc_config.get('VpcId')}")
print(f"Subnets: {vpc_config.get('SubnetIds', [])}")
print(f"Security Groups: {vpc_config.get('SecurityGroupIds', [])}")

# Remove VPC configuration
lambda_client.update_function_configuration(
    FunctionName='my-function',
    VpcConfig={
        'SubnetIds': [],
        'SecurityGroupIds': []
    }
)

# Security group rule for Lambda to access RDS
# Allow Lambda security group to access RDS on port 3306
import boto3

ec2 = boto3.client('ec2')

ec2.authorize_security_group_ingress(
    GroupId='sg-rds-12345678',  # RDS security group
    IpPermissions=[
        {
            'IpProtocol': 'tcp',
            'FromPort': 3306,
            'ToPort': 3306,
            'UserIdGroupPairs': [
                {
                    'GroupId': 'sg-lambda-12345678',  # Lambda security group
                    'Description': 'Allow Lambda to access RDS'
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
			ID:          312,
			Title:       "CloudFormation",
			Description: "Learn Infrastructure as Code with CloudFormation: templates, stacks, parameters, and best practices.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "CloudFormation Fundamentals",
					Content: `AWS CloudFormation provides a common language to model and provision AWS resources.

**CloudFormation Concepts:**
- **Template**: JSON or YAML file describing resources
- **Stack**: Collection of resources created from template
- **Parameters**: Input values for template
- **Outputs**: Values returned after stack creation
- **Resources**: AWS resources to create
- **Mappings**: Key-value pairs for conditional values

**Template Structure:**
- **AWSTemplateFormatVersion**: Template version
- **Description**: Template description
- **Parameters**: Input parameters
- **Mappings**: Key-value mappings
- **Conditions**: Conditional logic
- **Resources**: AWS resources (required)
- **Outputs**: Stack outputs

**Stack Operations:**
- **Create**: Create new stack
- **Update**: Update existing stack
- **Delete**: Delete stack and resources
- **Describe**: Get stack information
- **List**: List all stacks

**Benefits:**
- Infrastructure as Code
- Version control
- Repeatable deployments
- Dependency management
- Rollback capability

**Best Practices:**
- Use YAML for readability
- Parameterize templates
- Use nested stacks for complex infrastructure
- Validate templates before deployment
- Use change sets for updates
- Document templates`,
					CodeExamples: `# CloudFormation template example
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Simple EC2 instance'

Parameters:
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - t2.small
      - t2.medium
    Description: EC2 instance type

Resources:
  MyEC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: !Ref InstanceType
      KeyName: my-key-pair
      SecurityGroupIds:
        - !Ref MySecurityGroup
      Tags:
        - Key: Name
          Value: MyEC2Instance

  MySecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for EC2 instance
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

Outputs:
  InstanceId:
    Description: Instance ID
    Value: !Ref MyEC2Instance
    Export:
      Name: !Sub '\${AWS::StackName}-InstanceId'

  PublicIP:
    Description: Public IP address
    Value: !GetAtt MyEC2Instance.PublicIp

# Create stack
aws cloudformation create-stack \\
    --stack-name my-stack \\
    --template-body file://template.yaml \\
    --parameters ParameterKey=InstanceType,ParameterValue=t2.small

# Update stack
aws cloudformation update-stack \\
    --stack-name my-stack \\
    --template-body file://template.yaml \\
    --parameters ParameterKey=InstanceType,ParameterValue=t2.medium

# Create change set (preview changes)
aws cloudformation create-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set \\
    --template-body file://template.yaml

# Describe change set
aws cloudformation describe-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set

# Execute change set
aws cloudformation execute-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set

# Delete stack
aws cloudformation delete-stack --stack-name my-stack

# Validate template
aws cloudformation validate-template \\
    --template-body file://template.yaml

# Using boto3
import boto3

cf = boto3.client('cloudformation')

# Create stack
with open('template.yaml', 'r') as f:
    template_body = f.read()

cf.create_stack(
    StackName='my-stack',
    TemplateBody=template_body,
    Parameters=[
        {
            'ParameterKey': 'InstanceType',
            'ParameterValue': 't2.small'
        }
    ]
)

# Wait for stack creation
cf.get_waiter('stack_create_complete').wait(StackName='my-stack')

# Describe stack
response = cf.describe_stacks(StackName='my-stack')
print(response['Stacks'][0])`,
				},
				{
					Title: "CloudFormation Parameters and Mappings",
					Content: `Parameters allow templates to be reusable, and Mappings provide conditional values based on keys.

**Parameters:**
- Input values for templates
- Can have defaults, allowed values, constraints
- Can reference in Resources and Outputs
- Prompted during stack creation/update
- Can use AWS Systems Manager Parameter Store

**Parameter Types:**
- String, Number, List<Number>, CommaDelimitedList
- AWS-specific types (EC2::KeyPair::KeyName, etc.)
- SSM parameter types

**Parameter Properties:**
- Type: Parameter data type
- Default: Default value
- AllowedValues: List of allowed values
- AllowedPattern: Regex pattern
- MinLength/MaxLength: String length
- MinValue/MaxValue: Number range
- Description: Parameter description
- NoEcho: Hide value in console

**Mappings:**
- Key-value pairs for conditional values
- Lookup values based on keys
- Useful for region-specific values
- Can nest mappings
- Use Fn::FindInMap intrinsic function

**Use Cases:**
- Environment-specific values
- Region-specific AMIs
- Instance type selection
- Configuration variations

**Best Practices:**
- Use parameters for variable values
- Provide meaningful defaults
- Use constraints for validation
- Use mappings for region-specific values
- Document parameters`,
					CodeExamples: `# CloudFormation template with parameters
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Stack with parameters'

Parameters:
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - t2.small
      - t2.medium
    Description: EC2 instance type
    
  Environment:
    Type: String
    Default: dev
    AllowedValues:
      - dev
      - staging
      - prod
    Description: Environment name
    
  DatabasePassword:
    Type: String
    NoEcho: true
    MinLength: 8
    MaxLength: 128
    Description: Database password
    
  AllowedIP:
    Type: String
    AllowedPattern: '^([0-9]{1,3}\\.){3}[0-9]{1,3}/[0-9]{1,2}$'
    Description: Allowed IP CIDR block
    
  MultiAZ:
    Type: String
    Default: 'false'
    AllowedValues:
      - 'true'
      - 'false'
    Description: Enable Multi-AZ deployment

Mappings:
  RegionMap:
    us-east-1:
      AMI: ami-0c55b159cbfafe1f0
      S3Bucket: my-bucket-us-east-1
    us-west-2:
      AMI: ami-0abcdef1234567890
      S3Bucket: my-bucket-us-west-2
      
  EnvironmentMap:
    dev:
      InstanceType: t2.micro
      MinSize: 1
      MaxSize: 2
    prod:
      InstanceType: t2.medium
      MinSize: 2
      MaxSize: 10

Resources:
  MyEC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: !FindInMap [RegionMap, !Ref 'AWS::Region', AMI]
      InstanceType: !Ref InstanceType
      Tags:
        - Key: Environment
          Value: !Ref Environment

# Create stack with parameters
aws cloudformation create-stack \\
    --stack-name my-stack \\
    --template-body file://template.yaml \\
    --parameters \\
        ParameterKey=InstanceType,ParameterValue=t2.small \\
        ParameterKey=Environment,ParameterValue=prod \\
        ParameterKey=DatabasePassword,ParameterValue=MySecurePass123 \\
        ParameterKey=AllowedIP,ParameterValue=203.0.113.0/24

# Using boto3
import boto3

cf = boto3.client('cloudformation')

with open('template.yaml', 'r') as f:
    template_body = f.read()

cf.create_stack(
    StackName='my-stack',
    TemplateBody=template_body,
    Parameters=[
        {'ParameterKey': 'InstanceType', 'ParameterValue': 't2.small'},
        {'ParameterKey': 'Environment', 'ParameterValue': 'prod'},
        {'ParameterKey': 'DatabasePassword', 'ParameterValue': 'MySecurePass123'},
        {'ParameterKey': 'AllowedIP', 'ParameterValue': '203.0.113.0/24'}
    ]
)

# Get parameter values
stack = cf.describe_stacks(StackName='my-stack')['Stacks'][0]
for param in stack['Parameters']:
    if not param.get('ResolvedValue'):
        print(f"{param['ParameterKey']}: {param['ParameterValue']}")`,
				},
				{
					Title: "CloudFormation Nested Stacks",
					Content: `Nested Stacks allow you to break complex templates into smaller, reusable components.

**Nested Stacks:**
- Stack within a stack
- Parent stack references child stacks
- Child stacks created/deleted with parent
- Can pass parameters to nested stacks
- Can receive outputs from nested stacks

**Benefits:**
- Modularity and reusability
- Easier to manage complex infrastructure
- Share common components
- Separate concerns
- Version control friendly

**Nested Stack Structure:**
- Parent stack uses AWS::CloudFormation::Stack
- Template URL points to child template
- Parameters passed to child
- Outputs retrieved from child

**Template Location:**
- S3 bucket (recommended)
- Local file (for testing)
- Versioned templates
- Cross-region templates

**Use Cases:**
- Reusable network components
- Application components
- Environment-specific stacks
- Multi-tier applications

**Best Practices:**
- Store templates in S3
- Use versioning for templates
- Keep nested stacks focused
- Document dependencies
- Test nested stacks independently`,
					CodeExamples: `# Parent stack template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Parent stack with nested stacks'

Parameters:
  Environment:
    Type: String
    Default: dev

Resources:
  NetworkStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/my-bucket/templates/network.yaml
      Parameters:
        Environment: !Ref Environment
        VpcCidr: 10.0.0.0/16

  ApplicationStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/my-bucket/templates/application.yaml
      Parameters:
        Environment: !Ref Environment
        VpcId: !GetAtt NetworkStack.Outputs.VpcId
        SubnetIds: !GetAtt NetworkStack.Outputs.PublicSubnetIds

Outputs:
  ApplicationURL:
    Description: Application URL
    Value: !GetAtt ApplicationStack.Outputs.ApplicationURL
    Export:
      Name: !Sub '\${AWS::StackName}-ApplicationURL'

# Child stack: network.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Network resources'

Parameters:
  Environment:
    Type: String
  VpcCidr:
    Type: String
    Default: 10.0.0.0/16

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCidr
      Tags:
        - Key: Environment
          Value: !Ref Environment

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']

Outputs:
  VpcId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '\${AWS::StackName}-VpcId'
      
  PublicSubnetIds:
    Description: Public Subnet IDs
    Value: !Join [',', [!Ref PublicSubnet1]]
    Export:
      Name: !Sub '\${AWS::StackName}-PublicSubnetIds'

# Upload templates to S3
aws s3 cp network.yaml s3://my-bucket/templates/network.yaml
aws s3 cp application.yaml s3://my-bucket/templates/application.yaml
aws s3 cp parent.yaml s3://my-bucket/templates/parent.yaml

# Create parent stack
aws cloudformation create-stack \\
    --stack-name my-parent-stack \\
    --template-url https://s3.amazonaws.com/my-bucket/templates/parent.yaml \\
    --parameters ParameterKey=Environment,ParameterValue=prod

# Using boto3
import boto3

s3 = boto3.client('s3')
cf = boto3.client('cloudformation')

# Upload templates
s3.upload_file('network.yaml', 'my-bucket', 'templates/network.yaml')
s3.upload_file('application.yaml', 'my-bucket', 'templates/application.yaml')
s3.upload_file('parent.yaml', 'my-bucket', 'templates/parent.yaml')

# Create parent stack
cf.create_stack(
    StackName='my-parent-stack',
    TemplateURL='https://s3.amazonaws.com/my-bucket/templates/parent.yaml',
    Parameters=[
        {'ParameterKey': 'Environment', 'ParameterValue': 'prod'}
    ]
)

# Wait for stack creation
waiter = cf.get_waiter('stack_create_complete')
waiter.wait(StackName='my-parent-stack')

# List nested stacks
nested_stacks = cf.list_stack_resources(StackName='my-parent-stack')
for resource in nested_stacks['StackResourceSummaries']:
    if resource['ResourceType'] == 'AWS::CloudFormation::Stack':
        print(f"Nested Stack: {resource['PhysicalResourceId']}")`,
				},
				{
					Title: "CloudFormation Custom Resources",
					Content: `Custom Resources allow you to extend CloudFormation to perform actions not natively supported.

**Custom Resources:**
- Execute custom logic during stack operations
- Can call external APIs
- Can perform complex operations
- Use Lambda functions or SNS topics
- Return success or failure

**Custom Resource Lifecycle:**
- Create: When resource is created
- Update: When properties change
- Delete: When resource is deleted

**Custom Resource Properties:**
- ServiceToken: ARN of Lambda function or SNS topic
- Custom properties: Passed to handler

**Response Format:**
- Status: SUCCESS or FAILED
- PhysicalResourceId: Unique identifier
- Data: Key-value pairs for outputs
- Reason: Error message if failed
- NoEcho: Hide data in console

**Use Cases:**
- Third-party API integration
- Complex provisioning logic
- Data validation
- External system configuration
- Custom cleanup operations

**Best Practices:**
- Use Lambda for custom resources
- Implement idempotency
- Handle all lifecycle events
- Return appropriate status
- Log operations for debugging
- Set appropriate timeout`,
					CodeExamples: `# CloudFormation template with custom resource
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Stack with custom resource'

Resources:
  CustomResourceFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: my-custom-resource-handler
      Runtime: python3.11
      Handler: index.handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        ZipFile: |
          import json
          import boto3
          import cfnresponse
          
          def handler(event, context):
              try:
                  request_type = event['RequestType']
                  properties = event['ResourceProperties']
                  
                  if request_type == 'Create':
                      # Create resource
                      result = create_resource(properties)
                      cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                          'ResourceId': result['id'],
                          'Message': 'Resource created'
                      })
                  elif request_type == 'Update':
                      # Update resource
                      update_resource(event['PhysicalResourceId'], properties)
                      cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                          'Message': 'Resource updated'
                      })
                  elif request_type == 'Delete':
                      # Delete resource
                      delete_resource(event['PhysicalResourceId'])
                      cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
              except Exception as e:
                  cfnresponse.send(event, context, cfnresponse.FAILED, {
                      'Error': str(e)
                  })
          
          def create_resource(properties):
              # Custom creation logic
              return {'id': 'custom-resource-123'}
          
          def update_resource(resource_id, properties):
              # Custom update logic
              pass
          
          def delete_resource(resource_id):
              # Custom deletion logic
              pass

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  MyCustomResource:
    Type: AWS::CloudFormation::CustomResource
    Properties:
      ServiceToken: !GetAtt CustomResourceFunction.Arn
      CustomProperty1: value1
      CustomProperty2: value2

Outputs:
  CustomResourceOutput:
    Description: Output from custom resource
    Value: !GetAtt MyCustomResource.CustomOutput

# Lambda function for custom resource (Python)
import json
import boto3
import cfnresponse

def handler(event, context):
    try:
        request_type = event['RequestType']
        properties = event['ResourceProperties']
        
        if request_type == 'Create':
            # Perform creation
            result = create_external_resource(properties)
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                'PhysicalResourceId': result['id'],
                'CustomOutput': result['output']
            })
        elif request_type == 'Update':
            # Perform update
            update_external_resource(event['PhysicalResourceId'], properties)
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                'PhysicalResourceId': event['PhysicalResourceId']
            })
        elif request_type == 'Delete':
            # Perform deletion
            delete_external_resource(event['PhysicalResourceId'])
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        cfnresponse.send(event, context, cfnresponse.FAILED, {
            'Error': str(e)
        })

def create_external_resource(properties):
    # Call external API, create resource, etc.
    return {'id': 'external-resource-123', 'output': 'success'}

def update_external_resource(resource_id, properties):
    # Update external resource
    pass

def delete_external_resource(resource_id):
    # Delete external resource
    pass

# Using boto3
import boto3

cf = boto3.client('cloudformation')
lambda_client = boto3.client('lambda')

# Deploy Lambda function first
with open('custom_resource.zip', 'rb') as zip_file:
    lambda_client.create_function(
        FunctionName='my-custom-resource-handler',
        Runtime='python3.11',
        Role='arn:aws:iam::123456789012:role/lambda-execution-role',
        Handler='index.handler',
        Code={'ZipFile': zip_file.read()}
    )

# Create stack with custom resource
with open('template.yaml', 'r') as f:
    template_body = f.read()

cf.create_stack(
    StackName='my-stack',
    TemplateBody=template_body
)`,
				},
				{
					Title: "CloudFormation Best Practices",
					Content: `Following best practices ensures reliable, maintainable, and efficient CloudFormation templates.

**Template Organization:**
- Use YAML for readability
- Organize sections logically
- Use consistent naming
- Add descriptions
- Comment complex logic

**Parameterization:**
- Parameterize all variable values
- Use appropriate parameter types
- Provide meaningful defaults
- Add constraints for validation
- Use NoEcho for sensitive values

**Resource Management:**
- Use DeletionPolicy for critical resources
- Use UpdateReplacePolicy for replacement behavior
- Tag resources appropriately
- Use DependsOn for explicit dependencies
- Avoid circular dependencies

**Outputs and Exports:**
- Export important values
- Use meaningful output names
- Document outputs
- Use Fn::Sub for dynamic names
- Avoid exporting sensitive data

**Stack Management:**
- Use change sets for updates
- Validate templates before deployment
- Test in non-production first
- Use nested stacks for complexity
- Version control templates

**Security:**
- Don't hardcode credentials
- Use IAM roles and policies
- Enable encryption where possible
- Use parameter store for secrets
- Review IAM permissions

**Performance:**
- Minimize stack size
- Use nested stacks for large infrastructure
- Optimize template size
- Use intrinsic functions efficiently
- Avoid unnecessary resources

**Monitoring:**
- Monitor stack operations
- Set up CloudWatch alarms
- Review stack events
- Track resource changes
- Use CloudTrail for audit

**Best Practices Summary:**
- Version control all templates
- Use CI/CD for deployments
- Document architecture decisions
- Test thoroughly
- Review and update regularly
- Follow AWS Well-Architected Framework`,
					CodeExamples: `# Well-structured template example
AWSTemplateFormatVersion: '2010-09-09'
Description: |
  Production web application stack.
  Includes VPC, EC2 instances, RDS database, and load balancer.

Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
      - Label:
          default: Network Configuration
        Parameters:
          - VpcCidr
          - SubnetCidr
      - Label:
          default: Compute Configuration
        Parameters:
          - InstanceType
          - InstanceCount

Parameters:
  VpcCidr:
    Type: String
    Default: 10.0.0.0/16
    Description: CIDR block for VPC
    AllowedPattern: '^([0-9]{1,3}\\.){3}[0-9]{1,3}/[0-9]{1,2}$'
    
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues: [t2.micro, t2.small, t2.medium]
    Description: EC2 instance type

Conditions:
  CreateProdResources: !Equals [!Ref Environment, prod]
  UseMultiAZ: !Equals [!Ref MultiAZ, 'true']

Resources:
  VPC:
    Type: AWS::EC2::VPC
    DeletionPolicy: Retain
    Properties:
      CidrBlock: !Ref VpcCidr
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '\${AWS::StackName}-VPC'
        - Key: Environment
          Value: !Ref Environment

  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    UpdateReplacePolicy: Snapshot
    Condition: CreateProdResources
    Properties:
      DBInstanceIdentifier: !Sub '\${AWS::StackName}-db'
      Engine: mysql
      MultiAZ: !If [UseMultiAZ, true, false]
      BackupRetentionPeriod: 7
      Tags:
        - Key: Name
          Value: !Sub '\${AWS::StackName}-Database'

Outputs:
  VpcId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '\${AWS::StackName}-VpcId'
      
  DatabaseEndpoint:
    Description: RDS endpoint
    Value: !GetAtt Database.Endpoint.Address
    Condition: CreateProdResources

# Validate template
aws cloudformation validate-template \\
    --template-body file://template.yaml

# Create change set
aws cloudformation create-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set \\
    --template-body file://template.yaml \\
    --parameters ParameterKey=InstanceType,ParameterValue=t2.small

# Review changes
aws cloudformation describe-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set

# Execute change set
aws cloudformation execute-change-set \\
    --stack-name my-stack \\
    --change-set-name my-change-set

# Using boto3
import boto3

cf = boto3.client('cloudformation')

# Validate template
with open('template.yaml', 'r') as f:
    template_body = f.read()

try:
    cf.validate_template(TemplateBody=template_body)
    print("Template is valid")
except Exception as e:
    print(f"Template validation failed: {e}")

# Create change set
change_set = cf.create_change_set(
    StackName='my-stack',
    ChangeSetName='my-change-set',
    TemplateBody=template_body,
    Parameters=[
        {'ParameterKey': 'InstanceType', 'ParameterValue': 't2.small'}
    ],
    ChangeSetType='UPDATE'
)

# Wait for change set creation
waiter = cf.get_waiter('change_set_create_complete')
waiter.wait(
    StackName='my-stack',
    ChangeSetName='my-change-set'
)

# Describe changes
changes = cf.describe_change_set(
    StackName='my-stack',
    ChangeSetName='my-change-set'
)
for change in changes['Changes']:
    print(f"Action: {change['ResourceChange']['Action']}")
    print(f"Resource: {change['ResourceChange']['LogicalResourceId']}")

# Execute change set
cf.execute_change_set(
    StackName='my-stack',
    ChangeSetName='my-change-set'
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          313,
			Title:       "Auto Scaling and Load Balancing",
			Description: "Master Auto Scaling Groups and Elastic Load Balancing: scaling policies, health checks, and high availability.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Auto Scaling Groups",
					Content: `Auto Scaling Groups automatically maintain the desired number of EC2 instances.

**Auto Scaling Concepts:**
- **Auto Scaling Group (ASG)**: Group of EC2 instances
- **Launch Template/Configuration**: Instance configuration
- **Desired Capacity**: Target number of instances
- **Min Size**: Minimum instances
- **Max Size**: Maximum instances
- **Scaling Policy**: Rules for scaling

**Scaling Policies:**
- **Target Tracking**: Maintain target metric
- **Step Scaling**: Scale based on CloudWatch alarms
- **Simple Scaling**: Simple alarm-based scaling
- **Scheduled Scaling**: Scale at specific times

**Health Checks:**
- EC2 health checks
- ELB health checks (recommended)
- Replace unhealthy instances

**Use Cases:**
- Handle varying traffic
- Maintain availability
- Cost optimization
- Automatic recovery

**Best Practices:**
- Use launch templates
- Configure health checks properly
- Set appropriate min/max/desired
- Use multiple AZs
- Monitor scaling activities`,
					CodeExamples: `# Create launch template
aws ec2 create-launch-template \\
    --launch-template-name web-server-template \\
    --launch-template-data '{
        "ImageId": "ami-0c55b159cbfafe1f0",
        "InstanceType": "t2.micro",
        "SecurityGroupIds": ["sg-12345678"],
        "UserData": "IyEvYmluL2Jhc2gKc3VkbyB5dW0gaW5zdGFsbCAteSBuZ2lueA=="
    }'

# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \\
    --auto-scaling-group-name web-asg \\
    --launch-template LaunchTemplateName=web-server-template,Version='$Latest' \\
    --min-size 2 \\
    --max-size 10 \\
    --desired-capacity 2 \\
    --vpc-zone-identifier "subnet-12345678,subnet-87654321" \\
    --health-check-type ELB \\
    --health-check-grace-period 300 \\
    --target-group-arns arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-tg/1234567890123456

# Create scaling policy (target tracking)
aws autoscaling put-scaling-policy \\
    --auto-scaling-group-name web-asg \\
    --policy-name cpu-target-tracking \\
    --policy-type TargetTrackingScaling \\
    --target-tracking-configuration '{
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ASGAverageCPUUtilization"
        },
        "TargetValue": 50.0
    }'

# Create scaling policy (step scaling)
aws autoscaling put-scaling-policy \\
    --auto-scaling-group-name web-asg \\
    --policy-name step-scaling \\
    --policy-type StepScaling \\
    --adjustment-type ChangeInCapacity \\
    --step-adjustments '[{
        "MetricIntervalLowerBound": 0,
        "MetricIntervalUpperBound": 10,
        "ScalingAdjustment": 1
    }]' \\
    --cooldown 300

# Using boto3
import boto3

autoscaling = boto3.client('autoscaling')
ec2 = boto3.client('ec2')

# Create launch template
ec2.create_launch_template(
    LaunchTemplateName='web-server-template',
    LaunchTemplateData={
        'ImageId': 'ami-0c55b159cbfafe1f0',
        'InstanceType': 't2.micro',
        'SecurityGroupIds': ['sg-12345678']
    }
)

# Create Auto Scaling Group
autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='web-asg',
    LaunchTemplate={
        'LaunchTemplateName': 'web-server-template',
        'Version': '$Latest'
    },
    MinSize=2,
    MaxSize=10,
    DesiredCapacity=2,
    VPCZoneIdentifier='subnet-12345678,subnet-87654321',
    HealthCheckType='ELB',
    HealthCheckGracePeriod=300,
    TargetGroupARNs=['arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-tg/1234567890123456']
)

# Create target tracking policy
autoscaling.put_scaling_policy(
    AutoScalingGroupName='web-asg',
    PolicyName='cpu-target-tracking',
    PolicyType='TargetTrackingScaling',
    TargetTrackingConfiguration={
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'ASGAverageCPUUtilization'
        },
        'TargetValue': 50.0
    }
)`,
				},
				{
					Title: "Elastic Load Balancing",
					Content: `Elastic Load Balancing distributes incoming traffic across multiple targets for high availability.

**Load Balancer Types:**
- **Application Load Balancer (ALB)**: Layer 7, HTTP/HTTPS
- **Network Load Balancer (NLB)**: Layer 4, TCP/UDP
- **Classic Load Balancer (CLB)**: Legacy, not recommended

**ALB Features:**
- Path-based routing
- Host-based routing
- SSL/TLS termination
- Health checks
- Integration with Auto Scaling
- WebSocket support

**Target Groups:**
- EC2 instances
- IP addresses
- Lambda functions
- Containers (ECS/EKS)

**Health Checks:**
- Protocol and port
- Health check path
- Interval and timeout
- Healthy/unhealthy thresholds

**Best Practices:**
- Use ALB for HTTP/HTTPS
- Use NLB for TCP/UDP
- Enable cross-zone load balancing
- Configure appropriate health checks
- Use multiple AZs
- Enable access logs`,
					CodeExamples: `# Create Application Load Balancer
aws elbv2 create-load-balancer \\
    --name web-alb \\
    --subnets subnet-12345678 subnet-87654321 \\
    --security-groups sg-12345678 \\
    --scheme internet-facing \\
    --type application

# Create target group
aws elbv2 create-target-group \\
    --name web-targets \\
    --protocol HTTP \\
    --port 80 \\
    --vpc-id vpc-12345678 \\
    --health-check-protocol HTTP \\
    --health-check-path /health \\
    --health-check-interval-seconds 30 \\
    --health-check-timeout-seconds 5 \\
    --healthy-threshold-count 2 \\
    --unhealthy-threshold-count 3

# Register targets
aws elbv2 register-targets \\
    --target-group-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-targets/1234567890123456 \\
    --targets Id=i-1234567890abcdef0 Id=i-0987654321fedcba0

# Create listener
aws elbv2 create-listener \\
    --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/web-alb/1234567890123456 \\
    --protocol HTTP \\
    --port 80 \\
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-targets/1234567890123456

# Create HTTPS listener with certificate
aws elbv2 create-listener \\
    --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/web-alb/1234567890123456 \\
    --protocol HTTPS \\
    --port 443 \\
    --certificates CertificateArn=arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012 \\
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-targets/1234567890123456

# Create rule for path-based routing
aws elbv2 create-rule \\
    --listener-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:listener/app/web-alb/1234567890123456/1234567890123456 \\
    --priority 1 \\
    --conditions Field=path-pattern,Values='/api/*' \\
    --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/api-targets/1234567890123456

# Using boto3
import boto3

elbv2 = boto3.client('elbv2')

# Create load balancer
response = elbv2.create_load_balancer(
    Name='web-alb',
    Subnets=['subnet-12345678', 'subnet-87654321'],
    SecurityGroups=['sg-12345678'],
    Scheme='internet-facing',
    Type='application'
)
lb_arn = response['LoadBalancers'][0]['LoadBalancerArn']

# Create target group
tg_response = elbv2.create_target_group(
    Name='web-targets',
    Protocol='HTTP',
    Port=80,
    VpcId='vpc-12345678',
    HealthCheckProtocol='HTTP',
    HealthCheckPath='/health',
    HealthCheckIntervalSeconds=30,
    HealthCheckTimeoutSeconds=5,
    HealthyThresholdCount=2,
    UnhealthyThresholdCount=3
)
tg_arn = tg_response['TargetGroups'][0]['TargetGroupArn']

# Register targets
elbv2.register_targets(
    TargetGroupArn=tg_arn,
    Targets=[
        {'Id': 'i-1234567890abcdef0'},
        {'Id': 'i-0987654321fedcba0'}
    ]
)

# Create listener
elbv2.create_listener(
    LoadBalancerArn=lb_arn,
    Protocol='HTTP',
    Port=80,
    DefaultActions=[
        {
            'Type': 'forward',
            'TargetGroupArn': tg_arn
        }
    ]
)`,
				},
				{
					Title: "Launch Templates and Launch Configurations",
					Content: `Launch Templates and Launch Configurations define the configuration for instances launched by Auto Scaling Groups.

**Launch Templates:**
- Modern way to define instance configuration
- Versioned (can have multiple versions)
- Can specify instance type, AMI, security groups, etc.
- Supports instance metadata options
- Supports placement groups
- Recommended over Launch Configurations

**Launch Configurations:**
- Legacy way to define instance configuration
- Immutable (cannot modify)
- Must create new one to change
- Still supported but not recommended

**Launch Template Components:**
- AMI ID
- Instance type
- Key pair
- Security groups
- IAM role
- User data
- Block device mappings
- Network interfaces
- Tags

**Launch Template Versions:**
- Each version is immutable
- Can set default version
- Can use specific version or $Latest
- Can delete old versions

**Best Practices:**
- Use Launch Templates over Launch Configurations
- Version launch templates
- Use $Latest for flexibility
- Include all necessary configuration
- Test templates before using`,
					CodeExamples: `# Create launch template
aws ec2 create-launch-template \\
    --launch-template-name web-server-template \\
    --version-description "Initial version" \\
    --launch-template-data '{
        "ImageId": "ami-0c55b159cbfafe1f0",
        "InstanceType": "t2.micro",
        "KeyName": "my-key-pair",
        "SecurityGroupIds": ["sg-12345678"],
        "IamInstanceProfile": {
            "Name": "ec2-role"
        },
        "UserData": "IyEvYmluL2Jhc2gKc3VkbyB5dW0gaW5zdGFsbCAteSBuZ2lueA==",
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [{
                "Key": "Name",
                "Value": "WebServer"
            }]
        }]
    }'

# Create new version of launch template
aws ec2 create-launch-template-version \\
    --launch-template-name web-server-template \\
    --source-version 1 \\
    --version-description "Updated AMI" \\
    --launch-template-data '{
        "ImageId": "ami-0abcdef1234567890"
    }'

# Set default version
aws ec2 modify-launch-template \\
    --launch-template-name web-server-template \\
    --default-version 2

# Describe launch template
aws ec2 describe-launch-templates \\
    --launch-template-names web-server-template

# Describe launch template versions
aws ec2 describe-launch-template-versions \\
    --launch-template-name web-server-template

# Delete launch template version
aws ec2 delete-launch-template-versions \\
    --launch-template-name web-server-template \\
    --versions 1

# Delete launch template
aws ec2 delete-launch-template \\
    --launch-template-name web-server-template

# Using boto3
import boto3
import base64

ec2 = boto3.client('ec2')

# User data script
user_data_script = """#!/bin/bash
yum update -y
yum install -y nginx
systemctl start nginx
systemctl enable nginx
"""

user_data_encoded = base64.b64encode(user_data_script.encode()).decode()

# Create launch template
template = ec2.create_launch_template(
    LaunchTemplateName='web-server-template',
    VersionDescription='Initial version',
    LaunchTemplateData={
        'ImageId': 'ami-0c55b159cbfafe1f0',
        'InstanceType': 't2.micro',
        'KeyName': 'my-key-pair',
        'SecurityGroupIds': ['sg-12345678'],
        'IamInstanceProfile': {'Name': 'ec2-role'},
        'UserData': user_data_encoded,
        'TagSpecifications': [{
            'ResourceType': 'instance',
            'Tags': [
                {'Key': 'Name', 'Value': 'WebServer'},
                {'Key': 'Environment', 'Value': 'Production'}
            ]
        }]
    }
)
print(f"Launch Template ID: {template['LaunchTemplate']['LaunchTemplateId']}")

# Create new version
new_version = ec2.create_launch_template_version(
    LaunchTemplateName='web-server-template',
    SourceVersion='1',
    VersionDescription='Updated AMI',
    LaunchTemplateData={
        'ImageId': 'ami-0abcdef1234567890'
    }
)

# Set default version
ec2.modify_launch_template(
    LaunchTemplateName='web-server-template',
    DefaultVersion='2'
)

# Use launch template in Auto Scaling Group
autoscaling = boto3.client('autoscaling')
autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='web-asg',
    LaunchTemplate={
        'LaunchTemplateName': 'web-server-template',
        'Version': '$Latest'
    },
    MinSize=2,
    MaxSize=10,
    DesiredCapacity=2,
    VPCZoneIdentifier='subnet-12345678,subnet-87654321'
)`,
				},
				{
					Title: "Auto Scaling Policies and Metrics",
					Content: `Auto Scaling Policies define when and how to scale, using CloudWatch metrics to trigger scaling actions.

**Scaling Policy Types:**
- **Target Tracking**: Maintain target metric value
- **Step Scaling**: Scale based on CloudWatch alarms with steps
- **Simple Scaling**: Simple alarm-based scaling (legacy)
- **Scheduled Scaling**: Scale at specific times

**Target Tracking:**
- Simplest to configure
- Maintains target metric value
- Automatically creates CloudWatch alarms
- Supports: CPU, Network, Request count

**Step Scaling:**
- More control over scaling behavior
- Multiple scaling steps
- Based on CloudWatch alarms
- Can define different adjustments per step

**Simple Scaling:**
- Legacy policy type
- Single scaling adjustment
- Cooldown period required
- Not recommended for new deployments

**Scheduled Scaling:**
- Scale at specific times
- Useful for predictable patterns
- Can combine with other policies
- Timezone-aware

**Scaling Metrics:**
- CPU utilization
- Network in/out
- Request count per target
- Custom metrics
- ALB request count

**Best Practices:**
- Use Target Tracking for simplicity
- Use Step Scaling for complex scenarios
- Set appropriate cooldown periods
- Monitor scaling activities
- Test scaling policies
- Use multiple metrics when appropriate`,
					CodeExamples: `# Create target tracking policy
aws autoscaling put-scaling-policy \\
    --auto-scaling-group-name web-asg \\
    --policy-name cpu-target-tracking \\
    --policy-type TargetTrackingScaling \\
    --target-tracking-configuration '{
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ASGAverageCPUUtilization"
        },
        "TargetValue": 50.0,
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60
    }'

# Create step scaling policy
aws autoscaling put-scaling-policy \\
    --auto-scaling-group-name web-asg \\
    --policy-name step-scaling-cpu \\
    --policy-type StepScaling \\
    --adjustment-type ChangeInCapacity \\
    --step-adjustments '[{
        "MetricIntervalLowerBound": 0,
        "MetricIntervalUpperBound": 10,
        "ScalingAdjustment": 1
    }, {
        "MetricIntervalLowerBound": 10,
        "MetricIntervalUpperBound": 20,
        "ScalingAdjustment": 2
    }, {
        "MetricIntervalLowerBound": 20,
        "ScalingAdjustment": 3
    }]' \\
    --cooldown 300

# Create CloudWatch alarm for step scaling
aws cloudwatch put-metric-alarm \\
    --alarm-name cpu-high \\
    --alarm-description "CPU utilization is high" \\
    --metric-name CPUUtilization \\
    --namespace AWS/EC2 \\
    --statistic Average \\
    --period 60 \\
    --threshold 70 \\
    --comparison-operator GreaterThanThreshold \\
    --evaluation-periods 2 \\
    --alarm-actions arn:aws:autoscaling:us-east-1:123456789012:scalingPolicy:policy-id

# Create scheduled scaling action
aws autoscaling put-scheduled-update-group-action \\
    --auto-scaling-group-name web-asg \\
    --scheduled-action-name scale-up-morning \\
    --start-time "2024-01-01T08:00:00Z" \\
    --desired-capacity 5 \\
    --recurrence "0 8 * * *"

# Create scheduled scaling (scale down evening)
aws autoscaling put-scheduled-update-group-action \\
    --auto-scaling-group-name web-asg \\
    --scheduled-action-name scale-down-evening \\
    --start-time "2024-01-01T20:00:00Z" \\
    --desired-capacity 2 \\
    --recurrence "0 20 * * *"

# Describe scaling policies
aws autoscaling describe-policies \\
    --auto-scaling-group-name web-asg

# Delete scaling policy
aws autoscaling delete-policy \\
    --policy-name cpu-target-tracking \\
    --auto-scaling-group-name web-asg

# Using boto3
import boto3

autoscaling = boto3.client('autoscaling')
cloudwatch = boto3.client('cloudwatch')

# Create target tracking policy
autoscaling.put_scaling_policy(
    AutoScalingGroupName='web-asg',
    PolicyName='cpu-target-tracking',
    PolicyType='TargetTrackingScaling',
    TargetTrackingConfiguration={
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'ASGAverageCPUUtilization'
        },
        'TargetValue': 50.0,
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)

# Create step scaling policy
autoscaling.put_scaling_policy(
    AutoScalingGroupName='web-asg',
    PolicyName='step-scaling-cpu',
    PolicyType='StepScaling',
    AdjustmentType='ChangeInCapacity',
    StepAdjustments=[
        {
            'MetricIntervalLowerBound': 0,
            'MetricIntervalUpperBound': 10,
            'ScalingAdjustment': 1
        },
        {
            'MetricIntervalLowerBound': 10,
            'MetricIntervalUpperBound': 20,
            'ScalingAdjustment': 2
        },
        {
            'MetricIntervalLowerBound': 20,
            'ScalingAdjustment': 3
        }
    ],
    Cooldown=300
)

# Create CloudWatch alarm
cloudwatch.put_metric_alarm(
    AlarmName='cpu-high',
    AlarmDescription='CPU utilization is high',
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Statistic='Average',
    Period=60,
    Threshold=70,
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    AlarmActions=['arn:aws:autoscaling:us-east-1:123456789012:scalingPolicy:policy-id']
)

# Create scheduled scaling
autoscaling.put_scheduled_update_group_action(
    AutoScalingGroupName='web-asg',
    ScheduledActionName='scale-up-morning',
    StartTime='2024-01-01T08:00:00Z',
    DesiredCapacity=5,
    Recurrence='0 8 * * *'
)

# List scaling activities
activities = autoscaling.describe_scaling_activities(
    AutoScalingGroupName='web-asg'
)
for activity in activities['Activities']:
    print(f"Activity: {activity['Description']}, Status: {activity['StatusCode']}")`,
				},
				{
					Title: "Auto Scaling Lifecycle Hooks",
					Content: `Lifecycle Hooks allow you to perform custom actions when instances launch or terminate in Auto Scaling Groups.

**Lifecycle Hooks:**
- Execute actions during instance lifecycle
- Launch hooks: Before instance enters service
- Terminate hooks: Before instance terminates
- Can pause lifecycle transition
- Can complete or abandon hook

**Lifecycle States:**
- Pending: Instance launching
- Pending:Wait: In launch hook
- InService: Instance running
- Terminating: Instance terminating
- Terminating:Wait: In terminate hook
- Terminated: Instance terminated

**Use Cases:**
- Install software on launch
- Download configuration
- Register with load balancer
- Graceful shutdown
- Cleanup operations
- Health checks

**Hook Configuration:**
- Default result: CONTINUE or ABANDON
- Heartbeat timeout: How long to wait
- Notification target: SNS or SQS
- Role: IAM role for hook execution

**Best Practices:**
- Set appropriate timeout
- Handle hook failures gracefully
- Use SNS for notifications
- Implement idempotency
- Monitor hook execution
- Test hooks thoroughly`,
					CodeExamples: `# Create launch lifecycle hook
aws autoscaling put-lifecycle-hook \\
    --lifecycle-hook-name launch-hook \\
    --auto-scaling-group-name web-asg \\
    --lifecycle-transition autoscaling:EC2_INSTANCE_LAUNCHING \\
    --default-result CONTINUE \\
    --heartbeat-timeout 300 \\
    --notification-target-arn arn:aws:sns:us-east-1:123456789012:asg-notifications \\
    --role-arn arn:aws:iam::123456789012:role/AutoScalingNotificationRole

# Create terminate lifecycle hook
aws autoscaling put-lifecycle-hook \\
    --lifecycle-hook-name terminate-hook \\
    --auto-scaling-group-name web-asg \\
    --lifecycle-transition autoscaling:EC2_INSTANCE_TERMINATING \\
    --default-result CONTINUE \\
    --heartbeat-timeout 600 \\
    --notification-target-arn arn:aws:sns:us-east-1:123456789012:asg-notifications \\
    --role-arn arn:aws:iam::123456789012:role/AutoScalingNotificationRole

# Complete lifecycle hook action
aws autoscaling complete-lifecycle-action \\
    --lifecycle-hook-name launch-hook \\
    --auto-scaling-group-name web-asg \\
    --lifecycle-action-token token-here \\
    --lifecycle-action-result CONTINUE

# Abandon lifecycle hook action
aws autoscaling complete-lifecycle-action \\
    --lifecycle-hook-name terminate-hook \\
    --auto-scaling-group-name web-asg \\
    --lifecycle-action-token token-here \\
    --lifecycle-action-result ABANDON

# Record lifecycle action heartbeat
aws autoscaling record-lifecycle-action-heartbeat \\
    --lifecycle-hook-name launch-hook \\
    --auto-scaling-group-name web-asg \\
    --lifecycle-action-token token-here

# Describe lifecycle hooks
aws autoscaling describe-lifecycle-hooks \\
    --auto-scaling-group-name web-asg

# Delete lifecycle hook
aws autoscaling delete-lifecycle-hook \\
    --lifecycle-hook-name launch-hook \\
    --auto-scaling-group-name web-asg

# Lambda function to handle lifecycle hook
import json
import boto3
import time

def lambda_handler(event, context):
    sns_message = json.loads(event['Records'][0]['Sns']['Message'])
    
    lifecycle_action_token = sns_message['LifecycleActionToken']
    lifecycle_hook_name = sns_message['LifecycleHookName']
    auto_scaling_group_name = sns_message['AutoScalingGroupName']
    instance_id = sns_message['EC2InstanceId']
    lifecycle_transition = sns_message['LifecycleTransition']
    
    autoscaling = boto3.client('autoscaling')
    ec2 = boto3.client('ec2')
    
    try:
        if lifecycle_transition == 'autoscaling:EC2_INSTANCE_LAUNCHING':
            # Perform launch actions
            # Example: Install software, configure instance
            print(f"Configuring instance {instance_id}")
            # ... perform configuration ...
            
            # Complete lifecycle action
            autoscaling.complete_lifecycle_action(
                LifecycleHookName=lifecycle_hook_name,
                AutoScalingGroupName=auto_scaling_group_name,
                LifecycleActionToken=lifecycle_action_token,
                LifecycleActionResult='CONTINUE'
            )
            
        elif lifecycle_transition == 'autoscaling:EC2_INSTANCE_TERMINATING':
            # Perform termination actions
            # Example: Graceful shutdown, cleanup
            print(f"Shutting down instance {instance_id}")
            # ... perform cleanup ...
            
            # Complete lifecycle action
            autoscaling.complete_lifecycle_action(
                LifecycleHookName=lifecycle_hook_name,
                AutoScalingGroupName=auto_scaling_group_name,
                LifecycleActionToken=lifecycle_action_token,
                LifecycleActionResult='CONTINUE'
            )
    except Exception as e:
        print(f"Error handling lifecycle hook: {e}")
        # Abandon on error
        autoscaling.complete_lifecycle_action(
            LifecycleHookName=lifecycle_hook_name,
            AutoScalingGroupName=auto_scaling_group_name,
            LifecycleActionToken=lifecycle_action_token,
            LifecycleActionResult='ABANDON'
        )

# Using boto3
import boto3

autoscaling = boto3.client('autoscaling')

# Create launch hook
autoscaling.put_lifecycle_hook(
    LifecycleHookName='launch-hook',
    AutoScalingGroupName='web-asg',
    LifecycleTransition='autoscaling:EC2_INSTANCE_LAUNCHING',
    DefaultResult='CONTINUE',
    HeartbeatTimeout=300,
    NotificationTargetARN='arn:aws:sns:us-east-1:123456789012:asg-notifications',
    RoleARN='arn:aws:iam::123456789012:role/AutoScalingNotificationRole'
)

# Complete lifecycle action
autoscaling.complete_lifecycle_action(
    LifecycleHookName='launch-hook',
    AutoScalingGroupName='web-asg',
    LifecycleActionToken='token-here',
    LifecycleActionResult='CONTINUE'
)

# Record heartbeat (extend timeout)
autoscaling.record_lifecycle_action_heartbeat(
    LifecycleHookName='launch-hook',
    AutoScalingGroupName='web-asg',
    LifecycleActionToken='token-here'
)

# Describe hooks
hooks = autoscaling.describe_lifecycle_hooks(
    AutoScalingGroupName='web-asg'
)
for hook in hooks['LifecycleHooks']:
    print(f"Hook: {hook['LifecycleHookName']}, Transition: {hook['LifecycleTransition']}")`,
				},
				{
					Title: "Auto Scaling with Spot Instances",
					Content: `Auto Scaling Groups can use Spot Instances to reduce costs while maintaining availability.

**Spot Instances in ASG:**
- Mix On-Demand and Spot instances
- Significant cost savings (up to 90%)
- Automatic replacement on interruption
- Use for fault-tolerant workloads
- Configure allocation strategy

**Allocation Strategies:**
- **LowestPrice**: Lowest cost Spot pools
- **CapacityOptimized**: Best capacity availability
- **PriceCapacityOptimized**: Balance of price and capacity
- **Diversified**: Distribute across Spot pools

**Instance Types:**
- Can specify multiple instance types
- Increases Spot availability
- Diversifies interruption risk
- Use similar instance types

**Spot Instance Interruption:**
- 2-minute warning before interruption
- ASG automatically replaces instance
- Can use lifecycle hooks for cleanup
- Monitor interruption rates

**Use Cases:**
- Batch processing
- CI/CD workloads
- Data processing
- Web applications (with On-Demand mix)
- Development/testing

**Best Practices:**
- Mix On-Demand and Spot instances
- Use multiple instance types
- Use CapacityOptimized strategy
- Monitor Spot interruption rates
- Use lifecycle hooks for graceful handling
- Test interruption handling`,
					CodeExamples: `# Create launch template with multiple instance types
aws ec2 create-launch-template \\
    --launch-template-name spot-launch-template \\
    --launch-template-data '{
        "ImageId": "ami-0c55b159cbfafe1f0",
        "SecurityGroupIds": ["sg-12345678"],
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": "0.05",
                "SpotInstanceType": "one-time"
            }
        }
    }'

# Create ASG with mixed instances (On-Demand + Spot)
aws autoscaling create-auto-scaling-group \\
    --auto-scaling-group-name mixed-asg \\
    --launch-template LaunchTemplateName=web-server-template,Version='$Latest' \\
    --mixed-instances-policy '{
        "LaunchTemplate": {
            "LaunchTemplateSpecification": {
                "LaunchTemplateName": "web-server-template",
                "Version": "$Latest"
            },
            "Overrides": [
                {"InstanceType": "t3.micro"},
                {"InstanceType": "t3.small"},
                {"InstanceType": "t3.medium"}
            ]
        },
        "InstancesDistribution": {
            "OnDemandPercentageAboveBaseCapacity": 20,
            "SpotAllocationStrategy": "capacity-optimized",
            "SpotInstancePools": 2
        }
    }' \\
    --min-size 2 \\
    --max-size 10 \\
    --desired-capacity 4 \\
    --vpc-zone-identifier "subnet-12345678,subnet-87654321"

# Create ASG with Spot instances only
aws autoscaling create-auto-scaling-group \\
    --auto-scaling-group-name spot-asg \\
    --launch-template LaunchTemplateName=spot-launch-template,Version='$Latest' \\
    --mixed-instances-policy '{
        "LaunchTemplate": {
            "LaunchTemplateSpecification": {
                "LaunchTemplateName": "spot-launch-template",
                "Version": "$Latest"
            },
            "Overrides": [
                {"InstanceType": "t3.micro"},
                {"InstanceType": "t3.small"}
            ]
        },
        "InstancesDistribution": {
            "OnDemandPercentageAboveBaseCapacity": 0,
            "SpotAllocationStrategy": "capacity-optimized"
        }
    }' \\
    --min-size 2 \\
    --max-size 10 \\
    --desired-capacity 4 \\
    --vpc-zone-identifier "subnet-12345678,subnet-87654321"

# Using boto3
import boto3

autoscaling = boto3.client('autoscaling')
ec2 = boto3.client('ec2')

# Create launch template for Spot
ec2.create_launch_template(
    LaunchTemplateName='spot-launch-template',
    LaunchTemplateData={
        'ImageId': 'ami-0c55b159cbfafe1f0',
        'SecurityGroupIds': ['sg-12345678'],
        'InstanceMarketOptions': {
            'MarketType': 'spot',
            'SpotOptions': {
                'MaxPrice': '0.05',
                'SpotInstanceType': 'one-time'
            }
        }
    }
)

# Create ASG with mixed instances
autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='mixed-asg',
    LaunchTemplate={
        'LaunchTemplateName': 'web-server-template',
        'Version': '$Latest'
    },
    MixedInstancesPolicy={
        'LaunchTemplate': {
            'LaunchTemplateSpecification': {
                'LaunchTemplateName': 'web-server-template',
                'Version': '$Latest'
            },
            'Overrides': [
                {'InstanceType': 't3.micro'},
                {'InstanceType': 't3.small'},
                {'InstanceType': 't3.medium'}
            ]
        },
        'InstancesDistribution': {
            'OnDemandPercentageAboveBaseCapacity': 20,
            'SpotAllocationStrategy': 'capacity-optimized',
            'SpotInstancePools': 2
        }
    },
    MinSize=2,
    MaxSize=10,
    DesiredCapacity=4,
    VPCZoneIdentifier='subnet-12345678,subnet-87654321'
)

# Describe ASG instances
instances = autoscaling.describe_auto_scaling_instances()
for instance in instances['AutoScalingInstances']:
    if instance['AutoScalingGroupName'] == 'mixed-asg':
        print(f"Instance: {instance['InstanceId']}, Lifecycle: {instance['LifecycleState']}")

# Monitor Spot interruption
ec2_instances = ec2.describe_instances(
    Filters=[
        {'Name': 'tag:aws:autoscaling:groupName', 'Values': ['mixed-asg']}
    ]
)

for reservation in ec2_instances['Reservations']:
    for instance in reservation['Instances']:
        if instance.get('SpotInstanceRequestId'):
            print(f"Spot Instance: {instance['InstanceId']}")
            # Check for interruption notice
            try:
                # This would be done from within the instance
                # curl http://169.254.169.254/latest/meta-data/spot/instance-action
                pass
            except:
                pass`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          314,
			Title:       "Amazon API Gateway",
			Description: "Learn API Gateway: REST and HTTP APIs, integration with Lambda, authentication, and API management.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "API Gateway Fundamentals",
					Content: `Amazon API Gateway is a fully managed service for creating, publishing, and managing APIs.

**API Types:**
- **REST API**: RESTful API with full control
- **HTTP API**: Lower latency, lower cost, simplified
- **WebSocket API**: Real-time bidirectional communication

**Features:**
- Authentication and authorization
- Request/response transformation
- Throttling and rate limiting
- Caching
- Monitoring and logging
- API versioning

**Integration Types:**
- **Lambda**: Serverless function
- **HTTP**: HTTP endpoint
- **AWS Service**: Direct AWS service integration
- **Mock**: Return mock response

**Best Practices:**
- Use HTTP API when possible (lower cost)
- Implement authentication
- Set up throttling
- Enable caching
- Monitor API usage
- Use stages for environments`,
					CodeExamples: `# Create REST API
aws apigateway create-rest-api --name my-api

# Create HTTP API
aws apigatewayv2 create-api --name my-http-api --protocol-type HTTP

# Create resource and method
aws apigateway put-method --rest-api-id abc123 --resource-id def456 --http-method GET --authorization-type NONE

# Using boto3
import boto3
apigw = boto3.client('apigateway')
api = apigw.create_rest_api(name='my-api')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          315,
			Title:       "Amazon SNS and SQS",
			Description: "Learn SNS and SQS: messaging services, pub/sub, queues, and decoupled architectures.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "SNS and SQS Fundamentals",
					Content: `Amazon SNS (Simple Notification Service) and SQS (Simple Queue Service) enable decoupled, scalable messaging.

**SNS (Pub/Sub):**
- Topic-based messaging
- Multiple subscribers
- Push notifications
- Email, SMS, HTTP endpoints
- Fan-out pattern

**SQS (Queue):**
- Message queue service
- Decouple components
- Pull-based
- Standard and FIFO queues
- Long polling support

**Use Cases:**
- Event-driven architectures
- Microservices communication
- Notifications
- Task queues
- Decoupling systems

**Best Practices:**
- Use SNS for fan-out
- Use SQS for queuing
- Implement dead letter queues
- Set appropriate visibility timeout
- Monitor queue depth`,
					CodeExamples: `# Create SNS topic
aws sns create-topic --name my-topic

# Subscribe to topic
aws sns subscribe --topic-arn arn:aws:sns:us-east-1:123456789012:my-topic --protocol email --notification-endpoint user@example.com

# Create SQS queue
aws sqs create-queue --queue-name my-queue

# Send message
aws sqs send-message --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue --message-body "Hello"

# Using boto3
import boto3
sns = boto3.client('sns')
sqs = boto3.client('sqs')
topic = sns.create_topic(Name='my-topic')
queue = sqs.create_queue(QueueName='my-queue')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          316,
			Title:       "Amazon ECS",
			Description: "Learn ECS: container orchestration, tasks, services, clusters, and container management.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "ECS Fundamentals",
					Content: `Amazon ECS (Elastic Container Service) is a fully managed container orchestration service.

**ECS Components:**
- **Cluster**: Group of container instances
- **Task Definition**: Blueprint for containers
- **Task**: Running container instance
- **Service**: Maintains desired task count
- **Container**: Docker container

**Launch Types:**
- **EC2**: Run on EC2 instances
- **Fargate**: Serverless containers

**Features:**
- Auto Scaling
- Load balancing
- Service discovery
- Integration with CloudWatch
- Security groups and IAM

**Best Practices:**
- Use Fargate for simplicity
- Use task definitions for configuration
- Implement health checks
- Use service auto scaling
- Monitor container metrics`,
					CodeExamples: `# Create cluster
aws ecs create-cluster --cluster-name my-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service --cluster my-cluster --service-name my-service --task-definition my-task --desired-count 2

# Using boto3
import boto3
ecs = boto3.client('ecs')
cluster = ecs.create_cluster(clusterName='my-cluster')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          317,
			Title:       "Amazon EKS",
			Description: "Learn EKS: managed Kubernetes, clusters, nodes, pods, and Kubernetes on AWS.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "EKS Fundamentals",
					Content: `Amazon EKS (Elastic Kubernetes Service) is a managed Kubernetes service.

**EKS Features:**
- Managed Kubernetes control plane
- Compatible with Kubernetes
- Auto Scaling
- Integration with AWS services
- Security and compliance

**Components:**
- **Cluster**: Kubernetes cluster
- **Node Group**: EC2 instances running pods
- **Pod**: Smallest deployable unit
- **Service**: Stable network endpoint
- **Deployment**: Manages pods

**Best Practices:**
- Use managed node groups
- Implement RBAC
- Use IAM roles for service accounts
- Monitor cluster health
- Backup etcd data`,
					CodeExamples: `# Create EKS cluster
aws eks create-cluster --name my-cluster --role-arn arn:aws:iam::123456789012:role/eks-service-role --resources-vpc-config subnetIds=subnet-12345678

# Create node group
aws eks create-nodegroup --cluster-name my-cluster --nodegroup-name my-nodes --instance-types t3.medium --scaling-config minSize=1,maxSize=3

# Using boto3
import boto3
eks = boto3.client('eks')
cluster = eks.create_cluster(name='my-cluster', roleArn='arn:aws:iam::123456789012:role/eks-service-role', resourcesVpcConfig={'subnetIds': ['subnet-12345678']})`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          318,
			Title:       "AWS Systems Manager",
			Description: "Learn Systems Manager: parameter store, patch manager, session manager, and operations management.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Systems Manager Fundamentals",
					Content: `AWS Systems Manager provides operational insights and actions for AWS resources.

**Systems Manager Features:**
- Parameter Store: Secure configuration storage
- Patch Manager: Automated patching
- Session Manager: Secure shell access
- Run Command: Execute commands remotely
- State Manager: Maintain desired state
- Inventory: Resource inventory

**Use Cases:**
- Configuration management
- Patch management
- Remote access
- Automation
- Compliance

**Best Practices:**
- Use Parameter Store for secrets
- Automate patching
- Use Session Manager instead of SSH
- Document runbooks
- Monitor compliance`,
					CodeExamples: `# Put parameter
aws ssm put-parameter --name /myapp/db/password --value "Secret123" --type SecureString

# Get parameter
aws ssm get-parameter --name /myapp/db/password --with-decryption

# Start session
aws ssm start-session --target i-1234567890abcdef0

# Using boto3
import boto3
ssm = boto3.client('ssm')
ssm.put_parameter(Name='/myapp/db/password', Value='Secret123', Type='SecureString')`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
