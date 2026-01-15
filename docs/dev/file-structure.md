---
title: File Structure
---

# 📂 File Structure

```txt
project/
├── .github/                # GitHub specific files
|   ├── workflows/              # GitHub actions as workflows
|   └── release.yml             # Categories and labels for release notes
├── .vscode/                # VSCode specific files
|   ├── extensions.json         # Recommended extensions for the workspace
|   └── settings.json           # Common VSCode settings for the workspace (e.g. formatting, linting, etc...)
├── build/                  # Build files and directories (SHOULD NOT BE COMMITTED TO REPOSITORY)
├── dist/                   # Built distributions of this project (SHOULD NOT BE COMMITTED TO REPOSITORY)
├── docs/                   # Documentation of this project
|   ├── assets/                 # Any assets (images, audios, videos, js, css, html, etc...) used for the documentation
|   ├── diagrams/               # Diagrams related to this project
|   ├── blog/                   # Blog posts related to this project
|   └── .../                    # MkDocs pages - markdown files
├── examples/               # Example source codes of this project
├── requirements/           # Python dependency requirements for different environments
├── scripts/                # Helpful scripts to automate tasks or assist in the development process
├── site/                   # Built static site of the documentation (SHOULD NOT BE COMMITTED TO REPOSITORY)
├── src/                    # Source codes of this project
|   ├── modules/                # External modules for this project
|   |   ├── module_1/
|   |   ├── module_2/
|   |   └── .../
|   └── infer-ci/            # Main CODEBASE of this project as a python module
|       ├── __init__.py             # Initialize the module to be used as a package
|       ├── __version__.py          # Version of the module (should be updated and used with each release)
|       └── ...                     # Other main python files of this module
├── templates/              # Template files (if any, e.g. config files, etc...) used in this project
├── tests/                  # Tests for this project
|   ├── __init__.py             # Initialize the test module
|   ├── conftest.py             # Presets for pytest (e.g. fixtures, plugins, pre/post test hooks, etc...)
|   ├── test_1.py               # Test case files
|   ├── test_2.py
|   └── ...
├── __init__.py             # Initialize the whole project as a python module to import from other modules
├── .editorconfig           # Editor configuration for consistent coding styles for different editors
├── .env                    # Environment variables file (SHOULD NOT BE COMMITTED TO REPOSITORY)
├── .env.example            # Example environment variables file
├── .gitignore              # Files and directories to be ignored by git (e.g. data, models, results, etc...)
├── .markdownlint.json      # Markdown linting rules
├── .pre-commit-config.yaml # Pre-commit configuration file
├── CHANGELOG.md            # List of changes for each version of the project
├── environment.yml         # Conda environment file
├── LICENSE.txt             # License file for this project
├── Makefile                # Makefile for common commands and automation
├── MANIFEST.in             # Manifest file for setuptools (to include/exclude files in the source distribution)
├── mkdocs.yml              # MkDocs configuration file
├── pyproject.toml          # PEP 518 configuration file for python packaging
├── pytest.ini              # Pytest configuration file
├── README.md               # Main README file for this project
├── requirements.txt        # Main python dependency requirements for this project
├── setup.cfg               # Configuration for setuptools
└── setup.py                # Setup script for setuptools (for backward compatibility)
```
