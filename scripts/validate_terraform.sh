#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Terraform Validation Script
# EDCOCR -- Cloud-Native Deployment
#
# Runs terraform fmt -check and terraform validate across all modules.
# Exit code 0 = all pass, non-zero = one or more failures.
#
# Usage:
#   bash scripts/validate_terraform.sh
#
# Requirements:
#   - terraform CLI >= 1.5 must be on PATH
# -----------------------------------------------------------------------------

set -euo pipefail

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ---------------------------------------------------------------------------
# Helper: print result
# ---------------------------------------------------------------------------
print_result() {
    local label="$1"
    local status="$2"
    if [ "$status" = "PASS" ]; then
        echo -e "  [${GREEN}PASS${NC}] $label"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ "$status" = "FAIL" ]; then
        echo -e "  [${RED}FAIL${NC}] $label"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    elif [ "$status" = "SKIP" ]; then
        echo -e "  [${YELLOW}SKIP${NC}] $label"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Check terraform is installed
# ---------------------------------------------------------------------------
echo "==========================================="
echo "  Terraform Validation"
echo "==========================================="
echo ""

if ! command -v terraform &> /dev/null; then
    echo -e "${RED}ERROR: terraform CLI not found on PATH.${NC}"
    echo "Install from: https://developer.hashicorp.com/terraform/install"
    echo ""
    echo "Summary: 0 passed, 0 failed (terraform not available)"
    exit 1
fi

TF_VERSION=$(terraform version -json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['terraform_version'])" 2>/dev/null || terraform version | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
echo "Terraform version: $TF_VERSION"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Format check
# ---------------------------------------------------------------------------
echo "--- Format Check ---"

if terraform fmt -check -recursive "$TERRAFORM_DIR" > /dev/null 2>&1; then
    print_result "terraform fmt -check -recursive" "PASS"
else
    print_result "terraform fmt -check -recursive" "FAIL"
    echo "    Fix with: terraform fmt -recursive terraform/"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Validate each module
# ---------------------------------------------------------------------------
echo "--- Module Validation ---"

MODULES=(
    "modules/eks"
    "modules/gke"
    "modules/oke"
    "modules/shared"
)

for module in "${MODULES[@]}"; do
    module_path="$TERRAFORM_DIR/$module"
    module_label="$module"

    if [ ! -d "$module_path" ]; then
        print_result "$module_label" "SKIP"
        continue
    fi

    # Save and change directory
    pushd "$module_path" > /dev/null

    # Init with no backend
    if terraform init -backend=false -input=false > /dev/null 2>&1; then
        # Validate
        if terraform validate > /dev/null 2>&1; then
            print_result "$module_label" "PASS"
        else
            print_result "$module_label (validate)" "FAIL"
        fi
    else
        print_result "$module_label (init)" "FAIL"
    fi

    # Clean up .terraform directory created by init
    rm -rf .terraform .terraform.lock.hcl 2>/dev/null || true

    popd > /dev/null
done

echo ""

# ---------------------------------------------------------------------------
# Step 4: Validate environment compositions (format only, no init)
# ---------------------------------------------------------------------------
echo "--- Environment Checks ---"

ENVIRONMENTS=(
    "environments/staging"
    "environments/production"
)

for env in "${ENVIRONMENTS[@]}"; do
    env_path="$TERRAFORM_DIR/$env"
    env_label="$env"

    if [ ! -d "$env_path" ]; then
        print_result "$env_label" "SKIP"
        continue
    fi

    # Check tfvars.example exists
    if [ -f "$env_path/terraform.tfvars.example" ]; then
        print_result "$env_label/terraform.tfvars.example exists" "PASS"
    else
        print_result "$env_label/terraform.tfvars.example exists" "FAIL"
    fi
done

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "==========================================="
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  Summary: ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}, ${YELLOW}$SKIP_COUNT skipped${NC} ($TOTAL total)"
echo "==========================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi

exit 0
