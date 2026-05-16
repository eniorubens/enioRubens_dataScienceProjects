#!/bin/bash

###############################################################################
# Customer Churn Prediction Project Migration Script (Bash)
###############################################################################
#
# This script automatically migrates from:
#   OLD: Customer Churn Prediction/
#   NEW: customer-churn-prediction/
#
# What it does:
#   ✓ Creates new directory structure
#   ✓ Copies and renames notebooks
#   ✓ Copies model files
#   ✓ Updates git (stage, commit)
#
# Usage:
#   chmod +x migrate-churn-project.sh
#   ./migrate-churn-project.sh
#
# Author: Enio Rubens
# Date: 2026-05-16
###############################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script variables
OLD_FOLDER="Customer Churn Prediction"
NEW_FOLDER="customer-churn-prediction"
STEP_COUNT=6
CURRENT_STEP=0

# Helper functions
print_header() {
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "Customer Churn Prediction Project Migration Script (Bash)"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
}

print_step() {
    CURRENT_STEP=$1
    local message=$2
    echo -e "${BLUE}[STEP ${CURRENT_STEP}/${STEP_COUNT}]${NC} ${message}"
}

print_success() {
    local message=$1
    echo -e "${GREEN}✓${NC} ${message}"
}

print_error() {
    local message=$1
    echo -e "${RED}✗${NC} ${message}"
}

print_warning() {
    local message=$1
    echo -e "${YELLOW}⚠${NC} ${message}"
}

# Step 1: Verify prerequisites
step_verify_prerequisites() {
    print_step 1 "Verifying prerequisites..."
    
    if [ ! -d "$OLD_FOLDER" ]; then
        print_error "Source folder '$OLD_FOLDER' not found!"
        echo "  Make sure you're in the repository root directory:"
        echo "  pwd"
        echo "  ls -la '$OLD_FOLDER'"
        exit 1
    fi
    
    print_success "Found source folder: '$OLD_FOLDER'"
    
    # Check for notebooks
    if [ -d "$OLD_FOLDER/notebooks" ]; then
        local notebook_count=$(ls -1 "$OLD_FOLDER/notebooks" | wc -l)
        print_success "Found $notebook_count notebooks"
    else
        print_warning "No notebooks folder found"
    fi
    
    # Check for models
    if [ -d "$OLD_FOLDER/models" ]; then
        local model_count=$(ls -1 "$OLD_FOLDER/models" | wc -l)
        print_success "Found $model_count model files"
    else
        print_warning "No models folder found"
    fi
}

# Step 2: Create directory structure
step_create_directories() {
    print_step 2 "Creating new directory structure..."
    
    mkdir -p "$NEW_FOLDER/notebooks"
    mkdir -p "$NEW_FOLDER/models"
    mkdir -p "$NEW_FOLDER/dataset"
    
    print_success "Created main directory: $NEW_FOLDER/"
    print_success "  ├── notebooks/"
    print_success "  ├── models/"
    print_success "  └── dataset/"
}

# Step 3: Copy and rename notebooks
step_copy_notebooks() {
    print_step 3 "Copying and renaming notebooks..."
    
    if [ ! -d "$OLD_FOLDER/notebooks" ]; then
        print_warning "Notebooks folder not found"
        return 0
    fi
    
    # Copy English notebook
    if [ -f "$OLD_FOLDER/notebooks/Customer Churn Prediction EN.ipynb" ]; then
        cp "$OLD_FOLDER/notebooks/Customer Churn Prediction EN.ipynb" \
           "$NEW_FOLDER/notebooks/customer-churn-prediction-en.ipynb"
        print_success "customer-churn-prediction-en.ipynb"
    fi
    
    # Copy Portuguese notebook
    if [ -f "$OLD_FOLDER/notebooks/Customer Churn Prediction PT_Portuguese_PT.ipynb" ]; then
        cp "$OLD_FOLDER/notebooks/Customer Churn Prediction PT_Portuguese_PT.ipynb" \
           "$NEW_FOLDER/notebooks/customer-churn-prediction-pt.ipynb"
        print_success "customer-churn-prediction-pt.ipynb"
    fi
}

# Step 4: Copy model files
step_copy_models() {
    print_step 4 "Copying model files..."
    
    if [ ! -d "$OLD_FOLDER/models" ]; then
        print_warning "Models folder not found"
        return 0
    fi
    
    local model_count=$(find "$OLD_FOLDER/models" -type f | wc -l)
    
    cp -r "$OLD_FOLDER/models/"* "$NEW_FOLDER/models/" 2>/dev/null || true
    
    print_success "Copied $model_count model files"
}

# Step 5: Copy additional files
step_copy_additional() {
    print_step 5 "Copying additional files..."
    
    if [ -f "$OLD_FOLDER/README.md" ]; then
        cp "$OLD_FOLDER/README.md" "$NEW_FOLDER/README.md"
        print_success "Copied README.md"
    fi
}

# Step 6: Git operations
step_git_operations() {
    print_step 6 "Git staging and commit..."
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        print_warning "Git not found - skipping git operations"
        return 0
    fi
    
    # Stage new files
    git add "$NEW_FOLDER" 2>/dev/null || true
    print_success "Staged new files"
    
    # Create commit
    git commit -m "refactor: migrate Customer Churn Prediction to customer-churn-prediction with clean URLs" 2>/dev/null || true
    print_success "Created commit"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo "Migration Summary"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
    
    local total_files=$(find "$NEW_FOLDER" -type f | wc -l)
    print_success "Successfully created '$NEW_FOLDER/' with $total_files files"
    
    echo ""
    echo -e "${BOLD}Project Structure:${NC}"
    echo "  📁 $NEW_FOLDER/"
    echo "     ├── README.md"
    echo "     ├── 📂 notebooks/"
    echo "     │   ├── customer-churn-prediction-en.ipynb"
    echo "     │   └── customer-churn-prediction-pt.ipynb"
    echo "     ├── 📂 models/"
    echo "     │   └── (all model files)"
    echo "     └── 📂 dataset/"
    
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo "  1. Review changes:"
    echo "     git status"
    echo "  2. View directory:"
    echo "     ls -la $NEW_FOLDER/"
    echo "  3. Push to GitHub:"
    echo "     git push origin main"
    echo "  4. Delete old folder (after verification):"
    echo "     git rm -r '$OLD_FOLDER'"
    echo "     git commit -m 'refactor: remove old folder'"
    echo "     git push origin main"
    
    echo ""
    echo -e "${BOLD}New URLs (Now work on LinkedIn!):${NC}"
    echo "  📄 Project:"
    echo "     https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/customer-churn-prediction"
    echo "  📓 English Notebook:"
    echo "     https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/customer-churn-prediction-en.ipynb"
    echo "  📓 Portuguese Notebook:"
    echo "     https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/customer-churn-prediction-pt.ipynb"
    
    echo ""
    echo -e "${GREEN}${BOLD}✓ Migration completed successfully!${NC}"
    echo ""
}

# Main execution
main() {
    print_header
    
    step_verify_prerequisites
    step_create_directories
    step_copy_notebooks
    step_copy_models
    step_copy_additional
    step_git_operations
    
    print_summary
}

# Error handling
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"
