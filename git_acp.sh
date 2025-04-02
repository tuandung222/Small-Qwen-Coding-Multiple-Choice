#!/bin/bash
# Script to add, commit, and push changes to git repository

# Get commit message from command line argument, default to "Update" if not provided
MESSAGE=${1:-"Update"}

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

git add .
git commit -m "$MESSAGE"
git push
