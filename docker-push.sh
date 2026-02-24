#!/usr/bin/env bash
# ============================================================================
# docker-push.sh – Build roshambo2 Docker image and push to AWS ECR
#
# Usage:
#   ./docker-push.sh                   # uses defaults
#   ./docker-push.sh --tag v1.0.0      # custom tag
#   ./docker-push.sh --region us-west-2 --repo my-roshambo2
#
# Environment overrides (take precedence over flags):
#   AWS_ACCOUNT_ID   AWS account number
#   AWS_REGION        ECR region
#   ECR_REPO          ECR repository name
#   IMAGE_TAG         Docker image tag
# ============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_REGION="us-east-1"
DEFAULT_REPO="roshambo2"
DEFAULT_TAG="latest"

# ── Parse args ───────────────────────────────────────────────────────
REGION="${AWS_REGION:-$DEFAULT_REGION}"
REPO="${ECR_REPO:-$DEFAULT_REPO}"
TAG="${IMAGE_TAG:-$DEFAULT_TAG}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)  REGION="$2"; shift 2 ;;
        --repo)    REPO="$2";   shift 2 ;;
        --tag)     TAG="$2";    shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Resolve AWS account ─────────────────────────────────────────────
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${REPO}:${TAG}"

echo "============================================================"
echo "  Account:  ${ACCOUNT_ID}"
echo "  Region:   ${REGION}"
echo "  Repo:     ${REPO}"
echo "  Tag:      ${TAG}"
echo "  Image:    ${FULL_IMAGE}"
echo "============================================================"

# ── 1. Create ECR repo if it doesn't exist ───────────────────────────
if ! aws ecr describe-repositories \
        --region "${REGION}" \
        --repository-names "${REPO}" &>/dev/null; then
    echo "Creating ECR repository: ${REPO}"
    aws ecr create-repository \
        --region "${REGION}" \
        --repository-name "${REPO}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE
fi

# ── 2. Docker login to ECR ───────────────────────────────────────────
echo "Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

# ── 3. Build the image ───────────────────────────────────────────────
echo "Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build \
    --progress=plain \
    -t "${FULL_IMAGE}" \
    -t "${REPO}:${TAG}" \
    "${SCRIPT_DIR}"

# ── 4. Push to ECR ───────────────────────────────────────────────────
echo "Pushing to ECR..."
docker push "${FULL_IMAGE}"

echo ""
echo "============================================================"
echo "  ✓ Pushed: ${FULL_IMAGE}"
echo "============================================================"
echo ""
echo "Pull with:"
echo "  docker pull ${FULL_IMAGE}"
echo ""
echo "Run with:"
echo "  docker run --gpus all ${FULL_IMAGE} --help"
