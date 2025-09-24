#!/bin/bash

# -------------------------
# Restructure PV Forecast Repo
# -------------------------

echo "Starting repo restructuring..."

# Step 1: Create the clean structure
mkdir -p data/raw data/interim data/processed
mkdir -p notebooks
mkdir -p src/data src/features src/models src/training src/utils
mkdir -p models
mkdir -p experiments
mkdir -p reports
mkdir -p app

# Step 2: Move existing folders into the new structure
echo "Moving folders..."
mv ./data/* data/raw/ 2>/dev/null
mv ./src/data/* data/raw/ 2>/dev/null
mv ./models ./models_old 2>/dev/null
mv ./scripts ./src/training 2>/dev/null

# Step 3: Add __init__.py to each folder
echo "Adding __init__.py..."
find ./ -type d -exec touch {}/__init__.py \;

# Step 4: Git add, commit with messages
git add .
git commit -m "Restructure: create clean folder structure for PV forecasting project"

git add .
git commit -m "Restructure: move existing folders to appropriate locations"

git add .
git commit -m "Restructure: add __init__.py to all directories for Python packaging"

echo "Repo restructuring complete."
