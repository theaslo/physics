#!/bin/bash
# Remove any confusing local references
git branch -D development 2>/dev/null

# Fetch the latest from GitHub
git fetch origin

# Create a local development branch that properly tracks the remote
git checkout -b development origin/development

# Verify branches
git branch -vv
