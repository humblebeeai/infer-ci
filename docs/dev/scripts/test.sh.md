---
title: test.sh
---

# 🧪 test.sh

This script is used to run the pytest tests for the project.

The script performs the following operations:

- **Running pytest**: Runs the pytest tests for the project.
- **Logging**: If the `-l` or `--log` option is provided, the script will log the output of the pytest tests to console.
- **Coverage**: If the `-c` or `--cov` option is provided, the script will run the pytest tests with coverage.
- **Verbose**: If the `-v` or `--verbose` option is provided, the script will run the pytest tests with verbose error outputs.

**Usage**:

To execute the test script, simply run the following command in the terminal:

```sh
./test.sh [-l|--log] [-c|--cov] [-v|--verbose]
```

**Examples**:

- To test: `./test.sh`
- To test with logging: `./test.sh -l`
- To test with coverage: `./test.sh -c`
- To test with verbose: `./test.sh -v`
- To test with logging, coverage and verbose: `./test.sh -l -c -v`

This script will run the pytest tests for the project. It can also be used to run the tests with logging, coverage, and verbose options.

---

## References

- <https://docs.pytest.org>
