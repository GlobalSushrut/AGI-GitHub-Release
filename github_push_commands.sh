#!/bin/bash
# Replace the URL below with your actual GitHub repository URL after creation
# The format will be: https://github.com/yourusername/your-repo-name.git

# Add the GitHub repository as a remote
git remote add origin https://github.com/yourusername/AGI-GitHub-Release.git

# Verify the remote was added
git remote -v

# Push your code to GitHub
git push -u origin master

# After this, your code will be on GitHub
echo "Repository successfully pushed to GitHub!"
