***REMOVED***!/bin/bash
***REMOVED*** -------------------------
***REMOVED*** Create new branches with standard structure
***REMOVED*** -------------------------

BRANCHES=("data-pipeline-build" "pvlib-build" "rl-meta-build")

***REMOVED*** Base folder structure to create inside src for each branch
FOLDERS=("data" "features" "models" "training" "utils")

***REMOVED*** Function to create folders in src
create_structure() {
    for folder in "${FOLDERS[@]}"; do
        mkdir -p "src/$folder"
        touch "src/$folder/__init__.py"
    done
}

echo "Switching to main branch..."
git checkout main

for branch in "${BRANCHES[@]}"; do
    echo "Creating branch $branch..."
    git checkout -b "$branch"
    
    echo "Creating standard src folder structure in $branch..."
    create_structure
    
    echo "Adding changes to git..."
    git add src
    git commit -m "Initialize $branch with standard src structure"
    
    echo "Branch $branch setup complete."
    
    ***REMOVED*** Switch back to main to create the next branch
    git checkout main
done

echo "All branches created and initialized."