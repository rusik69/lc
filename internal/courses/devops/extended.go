package devops

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterDevOpsModules([]problems.CourseModule{
		// Intermediate Extended Modules (IDs 280-289, Order 30-39)
		{
			ID:          280,
			Title:       "GitLab CI/CD",
			Description: "GitLab CI/CD: pipelines, runners, and GitLab-native CI/CD workflows.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "GitLab CI/CD Fundamentals",
					Content: `GitLab provides built-in CI/CD capabilities with GitLab CI/CD.

**GitLab CI/CD Features:**
- **Pipelines**: Automated workflows
- **Jobs**: Individual tasks
- **Stages**: Group of jobs
- **Runners**: Execution agents
- **Variables**: Configuration values
- **Artifacts**: Build outputs
- **Caching**: Speed up builds

**Pipeline Structure:**
- Define in .gitlab-ci.yml
- Jobs run in stages
- Parallel execution within stages
- Conditional execution
- Manual approval gates

**GitLab Runners:**
- **Shared Runners**: GitLab.com provided
- **Specific Runners**: Project-specific
- **Group Runners**: Group-level
- **Instance Runners**: Self-hosted

**Best Practices:**
- Use .gitlab-ci.yml
- Cache dependencies
- Use artifacts
- Parallelize jobs
- Use variables for configuration
- Test pipelines locally`,
					CodeExamples: `# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - develop

test:
  stage: test
  image: node:18
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm test
    - npm run lint
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/app app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  environment:
    name: production
    url: https://app.example.com
  only:
    - main
  when: manual

# Using variables
variables:
  APP_VERSION: "1.0.0"
  NODE_VERSION: "18"

build:
  script:
    - echo "Building version $APP_VERSION"
    - npm install

# Conditional jobs
deploy:staging:
  script:
    - deploy.sh staging
  only:
    - develop
  except:
    - main

# Parallel jobs
test:unit:
  script:
    - npm run test:unit

test:integration:
  script:
    - npm run test:integration

# Artifacts and caching
build:
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          281,
			Title:       "CircleCI and Travis CI",
			Description: "Cloud CI/CD platforms: CircleCI and Travis CI workflows and best practices.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "CircleCI and Travis CI",
					Content: `Cloud-based CI/CD platforms for automated testing and deployment.

**CircleCI:**
- Cloud-hosted CI/CD
- Docker-based builds
- Parallel execution
- Workflow orchestration
- Caching and artifacts
- Orbs (reusable configs)

**Travis CI:**
- GitHub integration
- Multi-language support
- Matrix builds
- Deployment automation
- Environment variables
- Build matrix

**Key Features:**
- Easy GitHub integration
- Free tier available
- Docker support
- Parallel builds
- Caching
- Notifications

**Best Practices:**
- Use caching
- Parallelize tests
- Use matrix builds
- Set up notifications
- Use secrets management
- Optimize build times`,
					CodeExamples: `# CircleCI config.yml
version: 2.1

orbs:
  node: circleci/node@5.0.0
  docker: circleci/docker@2.0.0

jobs:
  build:
    docker:
      - image: cimg/node:18.0
    steps:
      - checkout
      - node/install-packages:
          pkg-manager: npm
      - run:
          name: Build
          command: npm run build
      - persist_to_workspace:
          root: .
          paths:
            - dist

  test:
    docker:
      - image: cimg/node:18.0
    steps:
      - checkout
      - node/install-packages:
          pkg-manager: npm
      - run:
          name: Test
          command: npm test
      - store_test_results:
          path: test-results

  deploy:
    docker:
      - image: cimg/base:stable
    steps:
      - attach_workspace:
          at: .
      - run:
          name: Deploy
          command: ./deploy.sh

workflows:
  version: 2
  build-test-deploy:
    jobs:
      - build
      - test:
          requires:
            - build
      - deploy:
          requires:
            - test
          filters:
            branches:
              only: main

# Travis CI .travis.yml
language: node_js
node_js:
  - "18"
  - "20"

cache:
  directories:
    - node_modules

install:
  - npm ci

script:
  - npm test
  - npm run lint

matrix:
  include:
    - node_js: "18"
      env: TEST_SUITE=unit
    - node_js: "20"
      env: TEST_SUITE=integration

deploy:
  provider: heroku
  app: myapp
  api_key:
    secure: $HEROKU_API_KEY
  on:
    branch: main`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          282,
			Title:       "Container Registries",
			Description: "Container registries: Docker Hub, ECR, GCR, ACR, and image management.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "Container Registry Management",
					Content: `Manage container images in registries across different platforms.

**Container Registries:**
- **Docker Hub**: Public registry
- **AWS ECR**: Amazon Elastic Container Registry
- **GCR**: Google Container Registry
- **ACR**: Azure Container Registry
- **GitHub Container Registry**: GitHub-native
- **Harbor**: Self-hosted registry

**Key Features:**
- Image storage
- Versioning
- Security scanning
- Access control
- Image replication
- Vulnerability scanning

**Best Practices:**
- Use private registries for production
- Tag images properly
- Scan for vulnerabilities
- Use image signing
- Implement access control
- Clean up old images
- Use multi-arch images`,
					CodeExamples: `# Docker Hub
docker login
docker tag myapp:latest username/myapp:1.0.0
docker push username/myapp:1.0.0

# AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker tag myapp:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:1.0.0
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:1.0.0

# GCR
gcloud auth configure-docker
docker tag myapp:latest gcr.io/my-project/myapp:1.0.0
docker push gcr.io/my-project/myapp:1.0.0

# ACR
az acr login --name myregistry
docker tag myapp:latest myregistry.azurecr.io/myapp:1.0.0
docker push myregistry.azurecr.io/myapp:1.0.0

# GitHub Container Registry
docker login ghcr.io -u USERNAME -p TOKEN
docker tag myapp:latest ghcr.io/username/myapp:1.0.0
docker push ghcr.io/username/myapp:1.0.0

# Image scanning (Trivy)
trivy image myapp:latest
trivy image --severity HIGH,CRITICAL myapp:latest

# Multi-arch build
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:1.0.0 --push .`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          283,
			Title:       "Puppet and Chef",
			Description: "Configuration management with Puppet and Chef: declarative infrastructure automation.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "Puppet and Chef Configuration Management",
					Content: `Agent-based configuration management tools for infrastructure automation.

**Puppet:**
- Declarative language
- Agent-based
- Puppet DSL
- Modules and manifests
- Facts and variables
- Hiera for data

**Chef:**
- Ruby-based DSL
- Agent-based
- Recipes and cookbooks
- Attributes and templates
- Chef Server
- Test Kitchen

**Key Concepts:**
- **Idempotency**: Same result every run
- **Declarative**: Describe desired state
- **Agent-based**: Agents on managed nodes
- **Pull Model**: Agents pull configuration
- **Modules/Cookbooks**: Reusable components

**Best Practices:**
- Use modules/cookbooks
- Test configurations
- Use version control
- Document changes
- Use environments
- Implement testing`,
					CodeExamples: `# Puppet manifest
# site.pp
node 'web1.example.com' {
  include nginx
  include mysql
}

# modules/nginx/manifests/init.pp
class nginx {
  package { 'nginx':
    ensure => installed,
  }
  
  service { 'nginx':
    ensure => running,
    enable => true,
    require => Package['nginx'],
  }
  
  file { '/etc/nginx/nginx.conf':
    ensure  => file,
    content => template('nginx/nginx.conf.erb'),
    notify  => Service['nginx'],
  }
}

# Chef recipe
# cookbooks/myapp/recipes/default.rb
package 'nginx' do
  action :install
end

service 'nginx' do
  action [:enable, :start]
end

template '/etc/nginx/nginx.conf' do
  source 'nginx.conf.erb'
  notifies :restart, 'service[nginx]'
end

# Chef attributes
# cookbooks/myapp/attributes/default.rb
default['myapp']['port'] = 80
default['myapp']['workers'] = 4`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          284,
			Title:       "SaltStack",
			Description: "SaltStack configuration management: remote execution and state management.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "SaltStack Fundamentals",
					Content: `SaltStack provides fast, scalable configuration management and remote execution.

**SaltStack Features:**
- **Salt Master**: Central control
- **Salt Minions**: Managed nodes
- **States**: Desired configuration
- **Grains**: System information
- **Pillars**: Secure data
- **Remote Execution**: Run commands

**Key Concepts:**
- **Master-Minion**: Centralized control
- **States**: Declarative configuration
- **Modules**: Execution modules
- **Formulas**: Reusable states
- **Grains**: System attributes
- **Pillars**: Secure variables

**Best Practices:**
- Use states for configuration
- Use formulas for reusability
- Secure pillars
- Test states
- Use environments`,
					CodeExamples: `# Salt state (nginx/init.sls)
nginx:
  pkg.installed

nginx_service:
  service.running:
    - name: nginx
    - enable: True
    - require:
      - pkg: nginx

/etc/nginx/nginx.conf:
  file.managed:
    - source: salt://nginx/files/nginx.conf
    - template: jinja
    - require:
      - pkg: nginx
    - watch_in:
      - service: nginx_service

# Remote execution
salt '*' cmd.run 'uptime'
salt 'web*' pkg.install nginx
salt '*' state.apply nginx`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          285,
			Title:       "CloudFormation and ARM Templates",
			Description: "Cloud-native Infrastructure as Code: AWS CloudFormation and Azure ARM templates.",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "CloudFormation and ARM Templates",
					Content: `Native cloud Infrastructure as Code for AWS and Azure.

**AWS CloudFormation:**
- JSON or YAML templates
- Stack management
- Change sets
- Drift detection
- Stack policies
- Nested stacks

**Azure ARM Templates:**
- JSON templates
- Resource groups
- Template functions
- Linked templates
- Parameters and variables
- Outputs

**Key Features:**
- Native cloud integration
- Stack/Resource group management
- Rollback capabilities
- Parameterization
- Template validation

**Best Practices:**
- Use YAML for readability
- Parameterize templates
- Use nested/linked templates
- Validate before deployment
- Use change sets
- Document templates`,
					CodeExamples: `# CloudFormation template (YAML)
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web server stack'

Parameters:
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - t2.small

Resources:
  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: !Ref InstanceType
      SecurityGroups:
        - !Ref WebServerSecurityGroup

  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Web server security group
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

Outputs:
  WebServerURL:
    Description: Web server URL
    Value: !GetAtt WebServer.PublicIp

# ARM Template
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmName": {
      "type": "string"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachines",
      "apiVersion": "2021-03-01",
      "name": "[parameters('vmName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "hardwareProfile": {
          "vmSize": "Standard_D2s_v3"
        }
      }
    }
  ]
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          286,
			Title:       "Pulumi",
			Description: "Pulumi: Infrastructure as Code using familiar programming languages.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Pulumi Infrastructure as Code",
					Content: `Pulumi enables Infrastructure as Code using real programming languages.

**Pulumi Features:**
- **Multi-language**: Python, TypeScript, Go, C#
- **Real Code**: Use full programming language
- **State Management**: Cloud-backed state
- **Preview**: See changes before applying
- **Import**: Import existing resources
- **Testing**: Test infrastructure code

**Key Concepts:**
- **Stack**: Deployment environment
- **Project**: Pulumi program
- **Resources**: Infrastructure components
- **State**: Current infrastructure state
- **Preview**: Dry-run changes

**Best Practices:**
- Use version control
- Use stacks for environments
- Preview before applying
- Test infrastructure code
- Use components for reusability`,
					CodeExamples: `# Pulumi (TypeScript)
import * as aws from "@pulumi/aws";

const vpc = new aws.ec2.Vpc("main", {
    cidrBlock: "10.0.0.0/16",
});

const subnet = new aws.ec2.Subnet("main", {
    vpcId: vpc.id,
    cidrBlock: "10.0.1.0/24",
});

const instance = new aws.ec2.Instance("web", {
    ami: "ami-0c55b159cbfafe1f0",
    instanceType: "t2.micro",
    subnetId: subnet.id,
});

export const instanceId = instance.id;

# Pulumi (Python)
import pulumi
import pulumi_aws as aws

vpc = aws.ec2.Vpc("main",
    cidr_block="10.0.0.0/16"
)

subnet = aws.ec2.Subnet("main",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24"
)

instance = aws.ec2.Instance("web",
    ami="ami-0c55b159cbfafe1f0",
    instance_type="t2.micro",
    subnet_id=subnet.id
)

pulumi.export("instance_id", instance.id)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          287,
			Title:       "Vagrant",
			Description: "Vagrant: development environment automation and VM management.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "Vagrant Development Environments",
					Content: `Vagrant automates development environment setup and management.

**Vagrant Features:**
- **Providers**: VirtualBox, VMware, Docker, etc.
- **Provisioners**: Shell, Ansible, Chef, Puppet
- **Boxes**: Base images
- **Networking**: Port forwarding, private networks
- **Synced Folders**: Share files with VM
- **Multi-VM**: Multiple VMs in one config

**Key Concepts:**
- **Vagrantfile**: Configuration file
- **Box**: Base image
- **Provider**: Virtualization platform
- **Provisioner**: Configuration tool
- **Synced Folders**: File sharing

**Best Practices:**
- Version control Vagrantfile
- Use provisioners
- Share Vagrantfiles
- Use boxes from Vagrant Cloud
- Document setup`,
					CodeExamples: `# Vagrantfile
Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"
  
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "2048"
    vb.cpus = 2
  end
  
  config.vm.network "forwarded_port", guest: 80, host: 8080
  config.vm.network "private_network", ip: "192.168.33.10"
  
  config.vm.synced_folder "./app", "/var/www/app"
  
  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y nginx
    systemctl start nginx
  SHELL
  
  config.vm.provision "ansible" do |ansible|
    ansible.playbook = "playbook.yml"
  end
end

# Multi-VM setup
Vagrant.configure("2") do |config|
  config.vm.define "web" do |web|
    web.vm.box = "ubuntu/jammy64"
    web.vm.network "private_network", ip: "192.168.33.10"
  end
  
  config.vm.define "db" do |db|
    db.vm.box = "ubuntu/jammy64"
    db.vm.network "private_network", ip: "192.168.33.20"
  end
end`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          288,
			Title:       "Packer",
			Description: "Packer: automated machine image building for multiple platforms.",
			Order:       38,
			Lessons: []problems.Lesson{
				{
					Title: "Packer Image Building",
					Content: `Packer creates identical machine images for multiple platforms.

**Packer Features:**
- **Multi-platform**: AWS, Azure, GCP, Docker, etc.
- **Provisioners**: Shell, Ansible, Chef, Puppet
- **Builders**: Platform-specific builders
- **Post-processors**: Image processing
- **Variables**: Parameterization
- **Templates**: JSON or HCL2

**Key Concepts:**
- **Builders**: Create images
- **Provisioners**: Configure images
- **Post-processors**: Process images
- **Templates**: Build configuration
- **Artifacts**: Output images

**Best Practices:**
- Use provisioners
- Parameterize templates
- Test builds
- Version images
- Use post-processors`,
					CodeExamples: `# Packer template (HCL2)
source "amazon-ebs" "ubuntu" {
  ami_name      = "myapp-{{timestamp}}"
  instance_type = "t2.micro"
  region        = "us-east-1"
  source_ami_filter {
    filters = {
      name                = "ubuntu/images/*ubuntu-jammy-22.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"]
  }
  ssh_username = "ubuntu"
}

build {
  name    = "myapp"
  sources = ["source.amazon-ebs.ubuntu"]
  
  provisioner "shell" {
    script = "setup.sh"
  }
  
  provisioner "ansible" {
    playbook_file = "playbook.yml"
  }
  
  post-processor "manifest" {
    output = "manifest.json"
  }
}

# Docker builder
source "docker" "ubuntu" {
  image  = "ubuntu:22.04"
  commit = true
}

build {
  sources = ["source.docker.ubuntu"]
  
  provisioner "shell" {
    inline = [
      "apt-get update",
      "apt-get install -y nginx"
    ]
  }
  
  post-processor "docker-tag" {
    repository = "myapp"
    tag        = ["latest", "1.0.0"]
  }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          289,
			Title:       "Helm",
			Description: "Helm: Kubernetes package manager for managing applications.",
			Order:       39,
			Lessons: []problems.Lesson{
				{
					Title: "Helm Kubernetes Package Manager",
					Content: `Helm simplifies Kubernetes application deployment and management.

**Helm Concepts:**
- **Charts**: Kubernetes packages
- **Releases**: Deployed chart instances
- **Repositories**: Chart storage
- **Values**: Configuration
- **Templates**: Kubernetes manifests
- **Hooks**: Lifecycle events

**Key Features:**
- Package management
- Versioning
- Dependency management
- Template engine
- Rollback capabilities
- Repository support

**Best Practices:**
- Use values files
- Version charts
- Test charts
- Use dependencies
- Document charts
- Use hooks appropriately`,
					CodeExamples: `# Chart structure
myapp/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    ingress.yaml

# Chart.yaml
apiVersion: v2
name: myapp
version: 1.0.0
description: My application
dependencies:
  - name: postgresql
    version: "12.0.0"
    repository: "https://charts.bitnami.com/bitnami"

# values.yaml
replicaCount: 3
image:
  repository: myapp
  tag: "1.0.0"
service:
  type: ClusterIP
  port: 80

# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Chart.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"

# Install chart
helm install myapp ./myapp
helm install myapp ./myapp -f values.prod.yaml
helm upgrade myapp ./myapp
helm rollback myapp 1
helm uninstall myapp`,
				},
			},
			ProblemIDs: []int{},
		},
		// Advanced Extended Modules (IDs 290-299, Order 40-49)
		{
			ID:          290,
			Title:       "Chaos Engineering",
			Description: "Chaos engineering: testing system resilience through controlled failures.",
			Order:       40,
			Lessons: []problems.Lesson{
				{
					Title: "Chaos Engineering Principles",
					Content: `Chaos engineering tests system resilience by introducing controlled failures.

**Chaos Engineering Principles:**
- **Hypothesis**: Form hypothesis about system behavior
- **Real-world Events**: Simulate real failure scenarios
- **Production**: Test in production (carefully)
- **Automate**: Automate chaos experiments
- **Minimize Blast Radius**: Limit impact
- **Measure**: Measure system response

**Chaos Experiments:**
- **Network Latency**: Introduce delays
- **Network Partition**: Split network
- **CPU Stress**: Overload CPU
- **Memory Stress**: Exhaust memory
- **Disk Failure**: Simulate disk issues
- **Pod Kill**: Kill containers
- **Service Failure**: Stop services

**Chaos Tools:**
- **Chaos Monkey**: Netflix's chaos tool
- **Chaos Mesh**: Kubernetes chaos engineering
- **Litmus**: Kubernetes chaos framework
- **Gremlin**: Chaos engineering platform
- **Chaos Toolkit**: Chaos experiments

**Best Practices:**
- Start small
- Test in staging first
- Have rollback plan
- Monitor closely
- Document experiments
- Learn from results`,
					CodeExamples: `# Chaos Mesh experiment
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-kill
spec:
  action: pod-failure
  mode: one
  selector:
    namespaces:
      - production
    labelSelectors:
      app: myapp
  duration: "5m"

# Network chaos
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay
spec:
  action: delay
  mode: one
  selector:
    namespaces:
      - production
  delay:
    latency: "100ms"
    correlation: "100"
    jitter: "0ms"
  duration: "10m"

# CPU stress
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: cpu-stress
spec:
  mode: one
  selector:
    namespaces:
      - production
  stressors:
    cpu:
      workers: 4
      load: 100
  duration: "5m"`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          291,
			Title:       "Distributed Tracing",
			Description: "Distributed tracing: Jaeger, Zipkin, and request tracing across microservices.",
			Order:       41,
			Lessons: []problems.Lesson{
				{
					Title: "Distributed Tracing with Jaeger and Zipkin",
					Content: `Distributed tracing tracks requests across microservices.

**Distributed Tracing Concepts:**
- **Trace**: Request journey
- **Span**: Individual operation
- **Parent-Child**: Span relationships
- **Tags**: Span metadata
- **Logs**: Span events
- **Baggage**: Context propagation

**Tracing Tools:**
- **Jaeger**: CNCF distributed tracing
- **Zipkin**: Distributed tracing system
- **OpenTelemetry**: Observability standard
- **Datadog APM**: Full-stack APM
- **New Relic**: Application monitoring

**Best Practices:**
- Instrument all services
- Use correlation IDs
- Sample appropriately
- Monitor trace volume
- Use structured logging
- Correlate with metrics`,
					CodeExamples: `# Jaeger instrumentation (Python)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

def handle_request():
    with tracer.start_as_current_span("handle_request") as span:
        span.set_attribute("user.id", user_id)
        with tracer.start_as_current_span("call_service_a"):
            call_service_a()
        with tracer.start_as_current_span("call_service_b"):
            call_service_b()

# Zipkin instrumentation
from py_zipkin.zipkin import zipkin_span

@zipkin_span(service_name='myapp', span_name='handle_request')
def handle_request():
    with zipkin_span(service_name='myapp', span_name='call_service'):
        call_service()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          292,
			Title:       "Observability",
			Description: "Observability: OpenTelemetry, metrics, logs, traces, and full-stack observability.",
			Order:       42,
			Lessons: []problems.Lesson{
				{
					Title: "Observability with OpenTelemetry",
					Content: `OpenTelemetry provides observability standards and instrumentation.

**Observability Pillars:**
- **Metrics**: Numerical measurements
- **Logs**: Event records
- **Traces**: Request journeys
- **Profiles**: Resource usage

**OpenTelemetry:**
- **Standard**: CNCF observability standard
- **Multi-language**: Many language SDKs
- **Vendor-neutral**: Works with any backend
- **Auto-instrumentation**: Automatic tracing
- **Exporters**: Send to any backend

**Best Practices:**
- Use OpenTelemetry
- Instrument all services
- Correlate metrics, logs, traces
- Use structured logging
- Set up dashboards
- Implement alerting`,
					CodeExamples: `# OpenTelemetry (Python)
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create metrics
request_counter = meter.create_counter(
    "http_requests_total",
    description="Total HTTP requests"
)

request_duration = meter.create_histogram(
    "http_request_duration_seconds",
    description="HTTP request duration"
)

# Use in application
def handle_request():
    with tracer.start_as_current_span("handle_request"):
        start_time = time.time()
        request_counter.add(1)
        # Handle request
        duration = time.time() - start_time
        request_duration.record(duration)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          293,
			Title:       "Serverless DevOps",
			Description: "Serverless DevOps: Lambda, Functions, and serverless application deployment.",
			Order:       43,
			Lessons: []problems.Lesson{
				{
					Title: "Serverless DevOps Practices",
					Content: `DevOps practices for serverless applications and functions.

**Serverless Platforms:**
- **AWS Lambda**: AWS serverless functions
- **Azure Functions**: Azure serverless
- **Google Cloud Functions**: GCP serverless
- **Vercel**: Frontend/serverless platform
- **Netlify**: JAMstack platform

**Serverless DevOps:**
- **Infrastructure**: Managed by platform
- **Deployment**: Function deployment
- **Monitoring**: Platform metrics
- **Scaling**: Automatic scaling
- **Cost**: Pay per use

**Best Practices:**
- Use Infrastructure as Code
- Implement CI/CD
- Monitor function metrics
- Optimize cold starts
- Use environment variables
- Implement proper error handling
- Test locally`,
					CodeExamples: `# AWS Lambda function
import json

def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

# Serverless Framework (serverless.yml)
service: myapp

provider:
  name: aws
  runtime: python3.9
  region: us-east-1

functions:
  hello:
    handler: handler.lambda_handler
    events:
      - http:
          path: hello
          method: get

# Terraform Lambda
resource "aws_lambda_function" "myapp" {
  filename         = "lambda.zip"
  function_name    = "myapp"
  role            = aws_iam_role.lambda_role.arn
  handler         = "handler.lambda_handler"
  runtime         = "python3.9"
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          294,
			Title:       "Edge Computing and DevOps",
			Description: "Edge computing DevOps: deploying and managing applications at the edge.",
			Order:       44,
			Lessons: []problems.Lesson{
				{
					Title: "Edge Computing DevOps",
					Content: `DevOps practices for edge computing deployments.

**Edge Computing:**
- **CDN**: Content delivery networks
- **Edge Functions**: Run code at edge
- **IoT**: Internet of Things
- **5G**: Next-gen networks
- **Edge Nodes**: Distributed compute

**Edge DevOps Challenges:**
- **Distribution**: Many locations
- **Connectivity**: Limited connectivity
- **Management**: Remote management
- **Updates**: Rolling updates
- **Monitoring**: Distributed monitoring

**Best Practices:**
- Use GitOps
- Implement over-the-air updates
- Monitor edge nodes
- Use edge-optimized images
- Implement health checks
- Plan for offline scenarios`,
					CodeExamples: `# Edge deployment (Kubernetes)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-app
spec:
  replicas: 100
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"

# Edge update strategy
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: edge-agent
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          295,
			Title:       "Database DevOps",
			Description: "Database DevOps: schema migrations, database CI/CD, and database automation.",
			Order:       45,
			Lessons: []problems.Lesson{
				{
					Title: "Database DevOps Practices",
					Content: `DevOps practices for database management and migrations.

**Database DevOps:**
- **Schema Migrations**: Version control schema
- **Database CI/CD**: Automate database changes
- **Backup Automation**: Automated backups
- **Testing**: Database testing
- **Rollback**: Schema rollback
- **Seeding**: Test data management

**Migration Tools:**
- **Flyway**: Database migrations
- **Liquibase**: Database change management
- **Alembic**: Python migrations
- **Sequelize**: Node.js migrations
- **Django Migrations**: Django ORM

**Best Practices:**
- Version control migrations
- Test migrations
- Backup before migrations
- Use transactions
- Implement rollback
- Review migrations`,
					CodeExamples: `# Flyway migration
-- V1__Create_users_table.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- V2__Add_username_to_users.sql
ALTER TABLE users ADD COLUMN username VARCHAR(100);

# Liquibase changelog
databaseChangeLog:
  - changeSet:
      id: 1
      author: devops
      changes:
        - createTable:
            tableName: users
            columns:
              - column:
                  name: id
                  type: INT
                  autoIncrement: true
                  constraints:
                    primaryKey: true
              - column:
                  name: email
                  type: VARCHAR(255)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          296,
			Title:       "Security Scanning and SAST/DAST",
			Description: "Security scanning: SAST, DAST, dependency scanning, and security automation.",
			Order:       46,
			Lessons: []problems.Lesson{
				{
					Title: "Security Scanning Tools",
					Content: `Automated security scanning in CI/CD pipelines.

**Security Scanning Types:**
- **SAST**: Static Application Security Testing
- **DAST**: Dynamic Application Security Testing
- **Dependency Scanning**: Check dependencies
- **Container Scanning**: Scan images
- **Infrastructure Scanning**: Scan IaC

**Security Tools:**
- **SonarQube**: Code quality and security
- **OWASP ZAP**: Web app security
- **Snyk**: Dependency scanning
- **Trivy**: Container scanning
- **Checkov**: Infrastructure scanning

**Best Practices:**
- Scan in CI/CD
- Fix high/critical issues
- Use multiple tools
- Automate scanning
- Review findings
- Track vulnerabilities`,
					CodeExamples: `# Snyk dependency scanning
snyk test
snyk monitor
snyk test --severity-threshold=high

# Trivy container scanning
trivy image myapp:latest
trivy image --severity HIGH,CRITICAL myapp:latest

# SonarQube scan
sonar-scanner \
  -Dsonar.projectKey=myapp \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://sonarqube:9000

# Checkov infrastructure scanning
checkov -d terraform/
checkov -f cloudformation.yaml`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          297,
			Title:       "Compliance and Auditing",
			Description: "Compliance and auditing: regulatory compliance, audit trails, and compliance automation.",
			Order:       47,
			Lessons: []problems.Lesson{
				{
					Title: "Compliance and Auditing",
					Content: `Ensure compliance and maintain audit trails in DevOps.

**Compliance Frameworks:**
- **SOC 2**: Security and availability
- **ISO 27001**: Information security
- **PCI DSS**: Payment card security
- **HIPAA**: Healthcare data
- **GDPR**: Data privacy

**Audit Requirements:**
- **Access Logs**: Who accessed what
- **Change Logs**: What changed
- **Configuration**: Current state
- **Compliance Reports**: Regular reports
- **Evidence**: Proof of compliance

**Best Practices:**
- Log all changes
- Implement access controls
- Regular audits
- Document procedures
- Use compliance tools
- Train teams`,
					CodeExamples: `# Audit logging
import logging

audit_logger = logging.getLogger('audit')
audit_logger.info('User accessed resource', extra={
    'user': user_id,
    'resource': resource_id,
    'action': 'read',
    'timestamp': datetime.now().isoformat()
})

# Compliance check script
#!/bin/bash
# Check for compliance violations

# Check for unencrypted storage
aws s3api list-buckets --query 'Buckets[?ServerSideEncryption==null]'

# Check for public access
aws s3api get-bucket-acl --bucket mybucket`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          298,
			Title:       "Performance Testing in CI/CD",
			Description: "Performance testing: load testing, stress testing, and performance testing automation.",
			Order:       48,
			Lessons: []problems.Lesson{
				{
					Title: "Performance Testing Automation",
					Content: `Integrate performance testing into CI/CD pipelines.

**Performance Testing Types:**
- **Load Testing**: Normal expected load
- **Stress Testing**: Beyond normal capacity
- **Spike Testing**: Sudden load increases
- **Endurance Testing**: Long duration
- **Volume Testing**: Large data volumes

**Performance Testing Tools:**
- **JMeter**: Load testing
- **Gatling**: Load testing framework
- **k6**: Modern load testing
- **Artillery**: Load testing toolkit
- **Locust**: Python load testing

**Best Practices:**
- Run in CI/CD
- Set performance thresholds
- Test regularly
- Monitor during tests
- Compare results
- Automate reporting`,
					CodeExamples: `# k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  let res = http.get('https://api.example.com/users');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}

# Gatling test
class BasicSimulation extends Simulation {
  val httpProtocol = http
    .baseUrl("https://api.example.com")
  
  val scn = scenario("Basic")
    .exec(http("request")
      .get("/users"))
    .pause(1)
  
  setUp(
    scn.inject(
      rampUsers(100) during (60 seconds)
    )
  ).protocols(httpProtocol)
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          299,
			Title:       "DevOps Metrics and KPIs",
			Description: "DevOps metrics and KPIs: measuring DevOps success and team performance.",
			Order:       49,
			Lessons: []problems.Lesson{
				{
					Title: "DevOps Metrics and KPIs",
					Content: `Measure DevOps success with key metrics and KPIs.

**DORA Metrics:**
- **Deployment Frequency**: How often deployments occur
- **Lead Time for Changes**: Time from commit to production
- **Mean Time to Recovery (MTTR)**: Time to recover from failures
- **Change Failure Rate**: Percentage of failed deployments

**Additional Metrics:**
- **Availability**: Uptime percentage
- **Throughput**: Work completed
- **Cycle Time**: Time to complete work
- **Error Rate**: Error percentage
- **Customer Satisfaction**: User satisfaction

**Best Practices:**
- Track DORA metrics
- Set targets
- Review regularly
- Use dashboards
- Share metrics
- Improve continuously`,
					CodeExamples: `# Deployment frequency
# Count deployments per day
SELECT DATE(created_at) as date, COUNT(*) as deployments
FROM deployments
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)

# Lead time
# Time from commit to production
SELECT 
  AVG(EXTRACT(EPOCH FROM (deployed_at - committed_at))) / 3600 as lead_time_hours
FROM deployments

# MTTR
# Mean time to recovery
SELECT 
  AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))) / 60 as mttr_minutes
FROM incidents

# Change failure rate
SELECT 
  COUNT(CASE WHEN status = 'failed' THEN 1 END) * 100.0 / COUNT(*) as failure_rate
FROM deployments`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
