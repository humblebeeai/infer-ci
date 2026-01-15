---
title: bump-version.sh
---

# 🏷 bump-version.sh

This script is used to manage the versioning of the project. It allows you to increment the major, minor, or patch part of the version, as per [Semantic Versioning](https://semver.org) rules.

The script carries out the following operations:

- **Loading environment variables**: If a `.env` file is present in the root directory, the script loads the environment variables from this file.
- **Sets variables**: Sets the `VERSION_FILE_PATH` and other variables. The `VERSION_FILE_PATH` variable is either loaded from the environment or defaults to `src/infer-ci/__version__.py`.
- **Parses input arguments**: It parses the `-b` or `--bump-type` argument for the type of version bump (**`major`**, **`minor`**, or **`patch`**) and `-p` or `--push-tag` to decide whether to push the tag to the Git repository or not.
- **Checks and increments the version**: It uses `get-version.sh` to extract the current version from the file specified by `VERSION_FILE_PATH`. Based on the bump type, it increments the appropriate part of the version and writes the new version back to the file.
- **Commits and tags**: If the `-p` or `--push-tag` flag was provided, it adds and commits the changes, pushes the changes, creates a new tag with the new version, and pushes the tag to the Git repository. It will prevent the operation if the tag already exists.

**Usage**:

To execute the bump version script, run the following command in the terminal:

```sh
./bump-version.sh -b=<bump_type> -p
```

Replace **`<bump_type>`** with either **`major`**, **`minor`**, or **`patch`** to indicate which part of the version to increment. The `-p` or `--push-tag` flag tells the script to commit the changes and push the tag to the Git repository.

**Examples**:

To bump the **`minor`** version and push the **new tag**, run:

```sh
./bump-version.sh -b=minor -p
```

This script streamlines the versioning process, reducing the chances of errors and ensuring consistency in versioning.

---

## References

- <https://semver.org>
