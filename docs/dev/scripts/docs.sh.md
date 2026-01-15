---
title: docs.sh
---

# 📝 docs.sh

This script is used to manage the documentation for the project, providing options to either serve a local documentation server or build the documentation as static HTML files.

The script performs the following operations:

- **Serving documentation**: If no flags are set, runs `mkdocs serve` to start a local documentation server for live preview.
- **Building documentation**: If the `-b` or `--build` flag is set, the script builds the documentation as static HTML files using `mkdocs build`, placing the output in the `site` directory.
- **Publishing documentation**: If the `-p` or `--publish` flag is set, the script can be extended to publish the documentation to GitHub Pages.

**Usage**:

To execute the documentation script, use the following command in the terminal:

```sh
./docs.sh [-b|--build] [-p|--publish]
```

**Examples**:

- To serve the documentation: `./docs.sh`
- To build the documentation: `./docs.sh -b`
- To publish the documentation: `./docs.sh -p`
