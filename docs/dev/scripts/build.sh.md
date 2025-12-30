---
title: build.sh
---

# 🏗️ build.sh

This script is used to build a Python project and optionally run tests and publish the package. It also includes a cleaning operation to clear the build directories.

This script has the following key features:

- **Checking for required tools**: It verifies if Python and the build package are installed on the system. If tests are not disabled, it also checks if pytest is installed. If uploading is enabled, it checks for the presence of Twine.
- **Command-line argument parsing**: It parses `-c` or `--disable-clean` to disable cleaning the build directories, `-t` or `--test` to enable running tests, `-u` or `--upload` to enable publishing the package, and `-p` or `--production` to switch the package repository from staging (default) to production.
- **Clean operation**: Cleans the build directories before and after building (if enabled). If `-c` or `--disable-clean` is passed, the script will not clean the build directories.
- **Testing operation**: Runs pytest tests if enabled by `-t` or `--test` flag.
- **Build operation**: Builds a Python package using the Python build package.
- **Publishing operation**: Publishes the built package to a PyPi repository using Twine if the `-u` or `--upload` flag is passed. Defaults to the TestPyPi (staging) repository, but can be switched to the production (PyPi) repository using the `-p` or `--production` flag.

**Usage**:

To execute the build script with the different flags, use the following commands:

```sh
./build.sh [-c|--disable-clean] [-t|--test] [-u|--upload] [-p|--production]
```

**Examples**:

- To build without cleaning: `./build.sh -c`
- To build with running tests: `./build.sh -t`
- To build and publish to the staging repository: `./build.sh -u`
- To build and publish to the production repository: `./build.sh -u -p`

This script is particularly beneficial for developers, streamlining the build, testing, and publishing process. It provides a one-stop solution for all the build needs of a Python project, reducing chances of errors and ensuring consistency.
