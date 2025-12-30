---
title: Publish Docs
---

# 📝 Publish Docs

## Overview

This GitHub Action automates the process of publishing documentation using **MkDocs** whenever changes are pushed to the `main` branch. It ensures that the latest documentation is deployed automatically.

## How It Works

The workflow is triggered when changes are made to:

- The `docs/` directory.
- The `mkdocs.yml` configuration file.

## Workflow Configuration

### **Trigger**

- **Triggered by:**
    - A push event to the `main` branch that modifies documentation files.

### **Jobs**

#### **1. Publish Docs**

This job builds and deploys the documentation using MkDocs.

- **Runs on:** `ubuntu-24.04`
- **Permissions:** `contents: write`
- **Steps:**
    1. **Checkout the repository** (with full history for proper deployment tracking).
    2. **Install dependencies** from `requirements/requirements.docs.txt`
    3. **Publish the documentation** using `mkdocs gh-deploy --force`

## Usage

This workflow runs automatically when changes are pushed to the `main` branch. However, you can manually deploy the documentation by running the following command locally:

```sh
mkdocs gh-deploy --force
```

## Environment Variables

- **`GITHUB_TOKEN`**: Used for authentication and deploying the documentation.

## Scripts Used

- **`mkdocs.yml`**: Configuration file for MkDocs.
- **Files in `docs/`**: Documentation content.

## Notes

- Ensure that `mkdocs` and required dependencies are correctly installed in `requirements.docs.txt`.
- The workflow ensures that documentation updates occur only when relevant files are modified.
- The changes are committed and published automatically to GitHub Pages.

## Troubleshooting

- If the workflow fails, ensure that `mkdocs` is properly installed and configured.
- If deployment issues occur, verify that GitHub Actions has the required `contents: write` permission.
- Check logs for any issues related to Git authentication or MkDocs execution.
