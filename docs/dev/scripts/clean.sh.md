---
title: clean.sh
---

# 🧹 clean.sh

This script is designed to clean up the build environment by removing artifacts and other temporary or unwanted files and directories.

The script performs the following operations:

- **Delete system files**: Finds and deletes all `.DS_Store` and `.Thumbs.db` files in the project directory and its subdirectories.
- **Delete cache directories**: Finds and deletes all `__pycache__` directories in the project directory and its subdirectories.
- **Delete project-related directories**: Removes directories created during the test and build process or by tools used in the project, such as `.benchmarks`, `.pytest_cache`, `build`, and `dist` directories.
- **Delete `.coverage` file**: Removes the `.coverage` file that's created when coverage information is collected for the project.

**Usage**:

To execute the clean script, simply run the following command in the terminal:

```sh
./clean.sh [-a|--all]
```

**Examples**:

- To clean just non-essential files: `./clean.sh`
- To clean all files: `./clean.sh -a`

This will clean up the project directory, removing any unnecessary files and directories and ensuring a clean environment for a fresh build.
