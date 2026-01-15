---
title: changelog.sh
---

# 📌 changelog.sh

This script automates updating both a `CHANGELOG.md` file and a `docs/release-notes.md` file using release information fetched from GitHub. It ensures consistent release documentation across the project and optionally commits and pushes changes to the Git repository.

The script performs the following operations:

- **Environment setup**:  
    - Runs from the project root.  
    - Loads environment variables from a `.env` file if present.  
- **Dependency checks**:  
    - Ensures the GitHub CLI (`gh`) is installed and authenticated.  
    - If `--commit` is specified, verifies that `git` is available.  
- **Variables setup**:  
    - `CHANGELOG_FILE_PATH` → Path to the changelog file (default: `./CHANGELOG.md`).  
    - `RELEASE_NOTES_FILE_PATH` → Path to the release notes file (default: `./docs/release-notes.md`).  
- **Input parsing**:  
    - `-c` or `--commit`: Commit changelog and release notes updates.  
    - `-p` or `--push`: Push updates to the remote repository (requires `-c`).  
- **Changelog update**:  
    - Fetches the latest release tag and body from GitHub (`gh release view`).  
    - Updates `CHANGELOG.md` with a new section for the latest release, including date and notes.  
- **Release notes update**:  
    - Updates `docs/release-notes.md` with a formatted entry for the latest release.  
    - Adds a YAML front matter block and header if the file does not already exist.  
- **Commit and push (optional)**:  
    - If `-c` is provided, stages and commits both updated files with a commit message.  
    - If `-p` is also provided, pushes the commit to the remote repository.

## Usage

To execute `changelog.sh`, run:

```sh
./changelog.sh [-c|--commit] [-p|--push]
```
