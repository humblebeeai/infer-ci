---
title: Build
---

# 🏗️ Build Python Package

To build the python package, run the following command:

```sh
# Install python build dependencies:
pip install -r ./requirements/requirements.build.txt

# Build python package:
python -m build
# Or use the build script:
./scripts/build.sh
```

## Build

```sh
# Install python build:
pip install -U build

# Build help:
python -m build --help
```

## References

- [Python Packaging User Guide](https://packaging.python.org)
- [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects)
- [Writing your `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml)
- [Setuptools Documentation](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)
- Blogs:
    - [Python Packaging Best Practices](https://medium.com/@miqui.ferrer/python-packaging-best-practices-4d6da500da5f)
    - [Generic Folder Structure for your Machine Learning Projects](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe)
    - [How to Upload your Python Package to PyPI](https://towardsdatascience.com/how-to-upload-your-python-package-to-pypi-de1b363a1b3)
