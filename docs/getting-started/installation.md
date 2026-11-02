---
title: Installation
---

# 🛠 Installation

## Requirements

- **Python 3.10 or newer** (`requires-python = ">=3.10,<4.0"`)
- **pip** (comes with Python)

Check your Python version:

```sh
python --version
```

[NOTE] On some systems (for example macOS), the command is `python3` instead of `python`. Use whichever prints a version of 3.10 or newer, and use the same command in the steps below.

## 1. 🧪 Create and activate a virtual environment

A virtual environment keeps infer-ci and its dependencies separate from other Python projects. Create one inside your project directory:

```sh
python -m venv .venv
```

Activate it:

```sh
# macOS / Linux:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv\Scripts\activate.bat
```

When the environment is active, your terminal prompt starts with `(.venv)`. Activate it again each time you open a new terminal. To leave the environment, run `deactivate`.

[TIP] **Conda users:** if your prompt shows `(base)`, you are in the conda base environment. Avoid installing into `base`. Either create the virtual environment above, or create a dedicated conda environment:

```sh
conda create -n infer-ci python=3.13
conda activate infer-ci
```

## 2. 📦 Install the package (users)

**OPTION A.** [**RECOMMENDED**] Install from **PyPI**:

```sh
pip install -U infer-ci
```

**OPTION B.** Install the latest development version directly from the **GitHub** repository:

```sh
pip install git+https://github.com/humblebeeai/infer-ci.git
```

## 3. ✅ Verify the installation

```sh
python -c "import infer_ci; print(infer_ci.__version__)"
```

If the installation succeeded, this prints the installed version (for example `0.2.1`). The package is installed as `infer-ci`, but it is imported in Python as `infer_ci` (with an underscore).

[NOTE] The first import can take a few seconds while Matplotlib builds its font cache. This happens only once.

Next, try the [Quick Start](quickstart.md).

---

## 🧑‍💻 Installation for developers

Use these options if you want to modify infer-ci, run its tests, or work from the source code. Create and activate a virtual environment first (see step 1).

### 1. 📥 Download or clone the repository

**1.1.** Prepare a projects directory (if it does not exist):

```sh
# Create projects directory:
mkdir -pv ~/workspaces/projects

# Enter into projects directory:
cd ~/workspaces/projects
```

**1.2.** Follow one of the options below **[A]**, **[B]** or **[C]**:

**OPTION A.** Clone the repository:

```sh
git clone https://github.com/humblebeeai/infer-ci.git && \
    cd infer-ci
```

**OPTION B.** Clone the repository (for **DEVELOPMENT**: git + ssh key):

```sh
git clone git@github.com:humblebeeai/infer-ci.git && \
    cd infer-ci
```

**OPTION C.** Download source code:

1. Download the archived **zip** file from [**releases**](https://github.com/humblebeeai/infer-ci/releases).
2. Extract it into the projects directory.

### 2. 📦 Install from the source code

[NOTE] Run these commands from the repository root (the directory that contains `pyproject.toml`). Choose one of the following methods **[A ~ D]**:

**OPTION A.** Install for the **DEVELOPMENT** environment (editable mode with test, build and docs dependencies):

```sh
pip install -e ".[dev]"

# Install pre-commit hooks:
pre-commit install
```

**OPTION B.** Install from the downloaded **source code**:

```sh
# Install directly from the source code:
pip install .

# Or install in editable mode:
pip install -e .
```

**OPTION C.** Install from **pre-built release** files:

1. Download the **`.whl`** or **`.tar.gz`** file from [**releases**](https://github.com/humblebeeai/infer-ci/releases).
2. Install with pip:

```sh
# Install from .whl file:
pip install ./infer_ci-[VERSION]-py3-none-any.whl

# Or install from .tar.gz file:
pip install ./infer_ci-[VERSION].tar.gz
```

**OPTION D.** Copy the **module** into the project directory (for **testing**):

```sh
# Install python dependencies:
pip install -r ./requirements.txt

# Copy the module source code into the project:
cp -r ./src/infer_ci [PROJECT_DIR]
# For example:
cp -r ./src/infer_ci /some/path/project/
```

**Staging builds.** To test a pre-release published to TestPyPI:

```sh
pip install -i https://test.pypi.org/simple -U infer-ci
```

Verify the installation the same way as in step 3 above.
