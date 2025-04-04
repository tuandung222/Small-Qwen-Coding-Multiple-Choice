#!/bin/bash
# Script to add, commit, and push changes to git repository

# Get the commit message from command line argument or use a default
COMMIT_MESSAGE=${1:-"Update QLoRA training documentation with optimizations and troubleshooting"}

# Run pre-commit hooks if installed
if command -v pre-commit &> /dev/null; then
    echo "Running pre-commit hooks..."
    pre-commit run --all-files
    if [ $? -ne 0 ]; then
        echo "Pre-commit hooks failed. Please fix the issues before committing."
        exit 1
    fi
else
    echo "pre-commit not installed. Skipping pre-commit hooks."
fi

# Add all changes
git add .

# Commit with message
git commit -m "$COMMIT_MESSAGE"

# Push to remote
git push origin main
#  || git push origin master

# Print status
echo "Git add, commit, and push completed"
echo "Commit message: $COMMIT_MESSAGE"
