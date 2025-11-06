 SageMaker vLLM Deployment

[![CI](https://github.com/YOUR_USERNAME/sagemaker-vllm-deployment/workflows/CI/badge.svg)](https://github.com/eun2ce/sagemaker-vllm-deployment/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Deploy any HuggingFace LLM to AWS SageMaker with vLLM acceleration. Solves Docker Schema2 compatibility issues and provides one-command deployment.

## Features

- **One-command deployment** - Deploy models to SageMaker in minutes
- **vLLM powered** - High-performance LLM inference with PagedAttention
- **Docker Schema2 fix** - Automatically handles compatibility issues
- **CI/CD ready** - GitHub Actions workflows included
- **Flexible** - Supports any HuggingFace model compatible with vLLM

## Prerequisites

- **macOS or Linux**
- **AWS CLI** - Configured with your credentials
- **Docker** - Docker Desktop (macOS) or Docker Engine (Linux)
- **skopeo** (macOS only) - Required for Docker Schema2 conversion
  brew install skopeo

## Quick Start

# 1) Create .env file
cd sagemaker-deployment
cp .env.example .env
# Edit .env with your actual values

# 2) Grant execution permissions to scripts
chmod +x scripts/*.sh

# 3) Build and push image
./scripts/build-and-push-image.sh

# 4) Deploy endpoint (use Image URI from build-and-push-image.sh output)
./scripts/deploy-endpoint.sh <ECR_IMAGE_URI>

# 5) Test endpoint
./scripts/test-endpoint.sh
# Or with custom input
SYSTEM="You are a coding assistant." USER="Write Python hello world." ./scripts/test-endpoint.sh


## GitHub Actions CI/CD Setup

### Prerequisites

1. **AWS IAM Role with OIDC** (Recommended)
   - Create IAM role with ECR and SageMaker permissions
   - Configure OIDC trust relationship with GitHub
   - See: [GitHub Actions OIDC Guide](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
   - See: [AWS OIDC Guide](https://aws.amazon.com/blogs/security/use-iam-roles-to-connect-github-actions-to-actions-in-aws/)

2. **GitHub Secrets Configuration**
   - Go to Repository Settings → Secrets and variables → Actions
   - Add the following secrets:

**Required Secrets:**
- `AWS_ROLE_ARN`: IAM Role ARN for OIDC authentication
- `AWS_ACCOUNT_ID`: Your AWS account ID (12 digits)
- `AWS_REGION`: AWS region (default: `ap-northeast-2`)
- `ECR_REPOSITORY`: ECR repository name

**Optional Secrets:**
- `DJL_BASE_REGISTRY`: DJL base image registry (default: `763104351884`)

### Workflows

- **CI**: Runs on push/PR to `main` branch
  - Docker image build test
  - Python linting
  - Shell script validation

- **CD**: Runs on push to `main` or tag `v*`
  - Builds and pushes Docker image to ECR
  - Tag strategy:
    - Git tags (`v*`) → Use tag name as image tag
    - Main branch → Use `latest` as image tag