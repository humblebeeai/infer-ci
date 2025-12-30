---
title: get-version.sh
---

# 🔍 get-version.sh

This script is used to retrieve the current version of the application from a specified version file.

The script performs the following operations:

- `VERSION_FILE_PATH` is either loaded from the environment or, if it's not present in the environment, it defaults to `src/infer-ci/__version__.py`.
- It first checks if the `VERSION_FILE_PATH` variable is not empty and if the file exists. If these conditions are met, it retrieves the value of `__version__` from the file by using `grep`, `awk`, and `tr` commands. The `grep` command filters the line containing `__version__ =` , the `awk` command splits the line into two parts at `=`, and the `tr` command removes the quotes around the version. If these operations fail, it exits the script with status code `2`.
- If the `VERSION_FILE_PATH` variable is empty or the file does not exist, it sets the current version to `0.0.0`.
- Finally, it echoes the current version to the console.

**Usage**:

To execute the get version script, simply run the following command in the terminal:

```sh
./get-version.sh
```

This script can be used to conveniently fetch the version. It is used by the `bump-version.sh` script to retrieve the current version before incrementing it.
