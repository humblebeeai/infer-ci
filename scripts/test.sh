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

if [ -z "$(which pytest)" ]; then
	echo "[ERROR]: 'pytest' not found or not installed!"
	exit 1
fi
## --- Base --- ##


## --- Variables --- ##
# Flags:
_IS_LOGGING=false
_IS_COVERAGE=false
_IS_VERBOSE=false
## --- Variables --- ##


## --- Main --- ##
main()
{
	## --- Menu arguments --- ##
	if [ -n "${1:-}" ]; then
		local _input
		for _input in "${@:-}"; do
			case ${_input} in
				-l | --log)
					_IS_LOGGING=true
					shift;;
				-c | --cov)
					_IS_COVERAGE=true
					shift;;
				-v | --verbose)
					_IS_VERBOSE=true
					shift;;
				*)
					echo "[ERROR]: Failed to parsing input -> ${_input}!"
					echo "[INFO]: USAGE: ${0}  -l, --log | -c, --cov | -v, --verbose"
					exit 1;;
			esac
		done
	fi
	## --- Menu arguments --- ##


	if [ "${_IS_COVERAGE}" == true ]; then
		if ! python -c "import pytest_cov" &> /dev/null; then
			echo "[ERROR]: 'pytest-cov' python package is not installed!"
			exit 1
		fi
	fi


	local _logging_param=""
	local _coverage_param=""
	local _verbose_param=""
	if [ "${_IS_LOGGING}" == true ]; then
		_logging_param="-o log_cli=true"
	fi

	if [ "${_IS_COVERAGE}" == true ]; then
		_coverage_param="--cov"
	fi

	if [ "${_IS_VERBOSE}" == true ]; then
		_verbose_param="-svv"
	fi

	echo "[INFO]: Running test..."
	# shellcheck disable=SC2086
	python -m pytest -v ${_coverage_param} ${_logging_param} ${_verbose_param} || exit 2
	echo "[OK]: Done."
}

main "${@:-}"
## --- Main --- ##
