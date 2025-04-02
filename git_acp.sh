#!/bin/bash
# Script to add, commit, and push changes to git repository

# Get commit message from command line argument, default to "Update" if not provided
MESSAGE=${1:-"Update"}

git add .
git commit -m "$MESSAGE"
git push
