---
title: diagrams.sh
---

# 🖼️ diagrams.sh

This script generates UML diagrams for a specified Python module using `pyreverse`. It checks for dependencies, loads environment variables, sets directories, and processes command-line arguments for customization of module names, directories, and output locations.

## Overview

The script performs the following operations:

- **Dependency Checks**: Verifies the presence of required tools:
    - `graphviz` for `.dot` file handling.
    - `python` for running Python-based commands.
    - `pylint` (with `pyreverse`) for generating UML diagrams.

- **Environment Variable Setup**: Sets default values for module name, module directory, and output directory (`MODULE_NAME`, `MODULE_DIR`, `OUTPUT_DIR`). These can be customized via environment variables or command-line arguments.

- **Argument Parsing**: Parses optional arguments to allow customization:
    - `-m` or `--module-name` to specify the module name.
    - `-d` or `--module-dir` to specify the module directory.
    - `-o` or `--output-dir` to specify the output directory.

- **Directory Creation**: Creates subdirectories within the output directory for organizing different types of UML and flowchart outputs:
    - `classes` for class diagrams.
    - `packages` for package diagrams.

- **Diagram Generation**: Runs `pyreverse` to create UML diagrams in multiple formats (`html`, `pdf`, `png`, and `svg`) and organizes them into respective directories.

- **Completion Message**: Displays a message confirming successful generation of diagrams.

## Usage

To run the script:

```sh
./diagrams.sh -m=<module_name> -d=<module_dir> -o=<output_dir>
```
