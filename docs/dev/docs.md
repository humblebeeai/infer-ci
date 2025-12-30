---
title: Docs
---

# 📝 Docs

To build the documentation, run the following command:

```sh
# Install python documentation dependencies:
pip install -r ./requirements/requirements.docs.txt

# Serve documentation locally (for development):
mkdocs serve
# Or use the docs script:
./scripts/docs.sh

# Or build documentation:
mkdocs build
# Or use the docs script:
./scripts/docs.sh -b
```

## Diagrams

Prerequisites:

- Install [Graphviz](https://graphviz.org/download)

To generate diagrams, run the following command:

```sh
# Install python documentation dependencies:
pip install -r ./requirements/requirements.docs.txt

# Generate diagrams:
./scripts/diagrams.sh
```

## MkDocs Material

### Installation

```sh
# Install mkdocs-material and mkdocstrings:
pip install -U mkdocs-material mkdocstrings[python]
```

### Commands

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.

### Docs layout

```txt
mkdocs.yml    # The configuration file.
docs/
    index.md  # The documentation homepage.
    ...       # Other markdown pages, images and other files.
```

## References

- [MkDocs Documentation](https://www.mkdocs.org)
- [MkDocs Material Documentation](https://squidfunk.github.io/mkdocs-material)
- [mkdocstrings Documentation](https://mkdocstrings.github.io)
