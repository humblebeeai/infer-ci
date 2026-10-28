---
title: 설치
---

# 🛠 설치

## 요구 사항

- **Python 3.10 이상** (`requires-python = ">=3.10,<4.0"`)
- **pip** (Python에 함께 설치됨)

Python 버전을 확인합니다.

```sh
python --version
```

[NOTE] macOS 등 일부 환경에서는 `python` 대신 `python3` 명령을 사용합니다. 3.10 이상의 버전이 출력되는 명령을 사용하고, 아래 단계에서도 같은 명령을 사용하세요.

## 1. 🧪 가상환경 만들기와 활성화

가상환경을 사용하면 infer-ci와 의존 패키지를 다른 Python 프로젝트와 분리해서 관리할 수 있습니다. 프로젝트 디렉터리에서 가상환경을 만듭니다.

```sh
python -m venv .venv
```

가상환경을 활성화합니다.

```sh
# macOS / Linux:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv\Scripts\activate.bat
```

가상환경이 활성화되면 터미널 프롬프트 앞에 `(.venv)`가 표시됩니다. 새 터미널을 열 때마다 다시 활성화해야 합니다. 가상환경을 종료하려면 `deactivate`를 실행합니다.

[TIP] **conda 사용자:** 프롬프트에 `(base)`가 표시되면 conda의 base 환경에 있는 상태입니다. base 환경에는 설치하지 않는 것이 좋습니다. 위 방법으로 가상환경을 만들거나, 다음과 같이 전용 conda 환경을 만드세요.

```sh
conda create -n infer-ci python=3.13
conda activate infer-ci
```

## 2. 📦 패키지 설치 (일반 사용자)

**방법 A.** [**권장**] **PyPI**에서 설치합니다.

```sh
pip install -U infer-ci
```

**방법 B.** **GitHub** 저장소에서 최신 개발 버전을 바로 설치합니다.

```sh
pip install git+https://github.com/humblebeeai/infer-ci.git
```

## 3. ✅ 설치 확인

```sh
python -c "import infer_ci; print(infer_ci.__version__)"
```

설치가 정상적으로 완료되면 설치된 버전(예: `0.2.1`)이 출력됩니다. 패키지 이름은 `infer-ci`(하이픈)이지만, Python에서 불러올 때는 `infer_ci`(밑줄)를 사용합니다.

[NOTE] 처음 불러올 때 Matplotlib이 글꼴 캐시를 만드느라 몇 초 걸릴 수 있습니다. 이 과정은 한 번만 진행됩니다.

다음으로 [빠른 시작](quickstart.ko.md)을 따라 해 보세요.

---

## 🧑‍💻 개발자용 설치

infer-ci를 수정하거나, 테스트를 실행하거나, 소스 코드로 작업하려면 아래 방법을 사용합니다. 먼저 가상환경을 만들고 활성화하세요(1단계 참고).

### 1. 📥 저장소 내려받기 또는 복제

**1.1.** 프로젝트 디렉터리를 준비합니다(없는 경우).

```sh
# Create projects directory:
mkdir -pv ~/workspaces/projects

# Enter into projects directory:
cd ~/workspaces/projects
```

**1.2.** 아래 **[A]**, **[B]**, **[C]** 중 한 가지 방법을 따릅니다.

**방법 A.** 저장소를 복제합니다.

```sh
git clone https://github.com/humblebeeai/infer-ci.git && \
    cd infer-ci
```

**방법 B.** 저장소를 복제합니다(**개발용**: git + SSH 키).

```sh
git clone git@github.com:humblebeeai/infer-ci.git && \
    cd infer-ci
```

**방법 C.** 소스 코드를 내려받습니다.

1. [**releases**](https://github.com/humblebeeai/infer-ci/releases)에서 **zip** 압축 파일을 내려받습니다.
2. 프로젝트 디렉터리에 압축을 풉니다.

### 2. 📦 소스 코드로 설치

[NOTE] 아래 명령은 저장소 최상위 디렉터리(`pyproject.toml`이 있는 디렉터리)에서 실행합니다. 다음 중 한 가지 방법을 선택하세요 **[A ~ D]**.

**방법 A.** **개발 환경**용으로 설치합니다(편집 가능 모드, 테스트·빌드·문서 의존성 포함).

```sh
pip install -e ".[dev]"

# Install pre-commit hooks:
pre-commit install
```

**방법 B.** 내려받은 **소스 코드**로 설치합니다.

```sh
# Install directly from the source code:
pip install .

# Or install in editable mode:
pip install -e .
```

**방법 C.** **미리 빌드된 릴리스** 파일로 설치합니다.

1. [**releases**](https://github.com/humblebeeai/infer-ci/releases)에서 **`.whl`** 또는 **`.tar.gz`** 파일을 내려받습니다.
2. pip로 설치합니다.

```sh
# Install from .whl file:
pip install ./infer_ci-[VERSION]-py3-none-any.whl

# Or install from .tar.gz file:
pip install ./infer_ci-[VERSION].tar.gz
```

**방법 D.** **모듈**을 프로젝트 디렉터리에 복사합니다(**테스트용**).

```sh
# Install python dependencies:
pip install -r ./requirements.txt

# Copy the module source code into the project:
cp -r ./src/infer_ci [PROJECT_DIR]
# For example:
cp -r ./src/infer_ci /some/path/project/
```

**스테이징 빌드.** TestPyPI에 배포된 사전 릴리스를 테스트하려면 다음 명령을 사용합니다.

```sh
pip install -i https://test.pypi.org/simple -U infer-ci
```

설치 확인은 위의 3단계와 같은 방법으로 합니다.
