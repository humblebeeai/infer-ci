
---
title: Installation
---

# 🛠 Installation

## 1. 📥 Download or clone the repository

Skip this step if you plan to install the package directly from PyPI or GitHub.

**1.1.** Create the projects directory if it does not exist:

```sh
mkdir -pv ~/workspaces/projects
cd ~/workspaces/projects
```

**1.2.** Choose one of the following options.

**OPTION A.** Clone the repository using HTTPS:

```sh
git clone https://github.com/humblebeeai/infer-ci.git && \
    cd infer-ci
```

**OPTION B.** Clone the repository using SSH (requires a configured SSH key):

```sh
git clone git@github.com:humblebeeai/infer-ci.git && \
    cd infer-ci
```

**OPTION C.** Download the source code:

1. Download the ZIP archive from [GitHub Releases](https://github.com/humblebeeai/infer-ci/releases).
2. Extract it into your projects directory.

## 2. 📦 Install the package

Choose one of the following options.

**OPTION A.** Install from PyPI:

```sh
# Install from TestPyPI
pip install -i https://test.pypi.org/simple -U infer-ci

# Install from PyPI
pip install -U infer-ci
```

**OPTION B.** Install directly from GitHub:

```sh
pip install git+https://github.com/humblebeeai/infer-ci.git
```

**OPTION C.** Install from the source code:

```sh
pip install .
```

For editable installation:

```sh
pip install -e .
```

**OPTION D.** Install for development:

```sh
pip install -e ".[dev]"
pre-commit install
```

**OPTION E.** Install from a release package:

1. Download a `.whl` or `.tar.gz` file from [GitHub Releases](https://github.com/humblebeeai/infer-ci/releases).
2. Install the downloaded file:

```sh
pip install ./infer-ci-[VERSION]-py3-none-any.whl
```

Or:

```sh
pip install ./infer-ci-[VERSION].tar.gz
```

**OPTION F.** Copy the module into another project for testing:

```sh
pip install -r ./requirements.txt
cp -r ./src/infer-ci [PROJECT_DIR]
```

Replace `[PROJECT_DIR]` with your project directory.
