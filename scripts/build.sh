#!/bin/bash
set -euo pipefail


## --- Base --- ##
# Getting path of this script file:
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_PROJECT_DIR="$(cd "${_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${_PROJECT_DIR}" || exit 2


if [ -z "$(which python)" ]; then
	echo "[ERROR]: 'python' not found or not installed!"
	exit 1
fi

if ! python -c "import build" &> /dev/null; then
	echo "[ERROR]: 'build' python package is not installed!"
	exit 1
fi
## --- Base --- ##


## --- Variables --- ##
# Flags:
_IS_CLEAN=true
_IS_TEST=false
_IS_UPLOAD=false
_IS_STAGING=true
## --- Variables --- ##


## --- Main --- ##
main()
{
	## --- Menu arguments --- ##
	if [ -n "${1:-}" ]; then
		local _input
		for _input in "${@:-}"; do
			case ${_input} in
				-c | --disable-clean)
					_IS_CLEAN=false
					shift;;
				-t | --test)
					_IS_TEST=true
					shift;;
				-u | --upload)
					_IS_UPLOAD=true
					shift;;
				-p | --production)
					_IS_STAGING=false
					shift;;
				*)
					echo "[ERROR]: Failed to parsing input -> ${_input}!"
					echo "[INFO]: USAGE: ${0}  -c, --disable-clean | -t, --test | -u, --upload | -p, --production"
					exit 1;;
			esac
		done
	fi
	## --- Menu arguments --- ##


	if [ "${_IS_UPLOAD}" == true ]; then
		if [ -z "$(which twine)" ]; then
			echo "[ERROR]: 'twine' not found or not installed!"
			exit 1
		fi
	fi


	if [ "${_IS_CLEAN}" == true ]; then
		./scripts/clean.sh || exit 2
	fi

	if [ "${_IS_TEST}" == true ]; then
		./scripts/test.sh || exit 2
	fi


	echo "[INFO]: Building package..."
	# python setup.py sdist bdist_wheel || exit 2
	python -m build || exit 2
	echo "[OK]: Done."

	if [ "${_IS_UPLOAD}" == true ]; then
		echo "[INFO]: Publishing package..."
		python -m twine check dist/* || exit 2
		if [ "${_IS_STAGING}" == true ]; then
			python -m twine upload --repository testpypi dist/* || exit 2
		else
			python -m twine upload dist/* || exit 2
		fi
		echo "[OK]: Done."

		if [ "${_IS_CLEAN}" == true ]; then
			./scripts/clean.sh || exit 2
		fi
	fi
}

main "${@:-}"
## --- Main --- ##
