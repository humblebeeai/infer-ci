#!/bin/bash
set -euo pipefail


## --- Base --- ##
# Getting path of this script file:
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_PROJECT_DIR="$(cd "${_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${_PROJECT_DIR}" || exit 2


# Loading .env file (if exists):
if [ -f ".env" ]; then
	# shellcheck disable=SC1091
	source .env
fi


if [ -z "$(which dot)" ]; then
	echo "[ERROR]: 'graphiz' not found or not installed!"
	exit 1
fi

if [ -z "$(which python)" ]; then
	echo "[ERROR]: 'python' not found or not installed!"
	exit 1
fi

if [ -z "$(which pyreverse)" ]; then
	echo "[ERROR]: 'pylint' not found or not installed!"
	exit 1
fi
## --- Base --- ##


## --- Variables --- ##
# Load from envrionment variables:
MODULE_NAME="${MODULE_NAME:-infer-ci}"
MODULE_DIR="${MODULE_DIR:-./src/${MODULE_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-./docs/diagrams}"


_MODULE_NAME=""
_MODULE_DIR=""
_OUTPUT_DIR=""
## --- Variables --- ##


## --- Main --- ##
main()
{
	## --- Menu arguments --- ##
	if [ -n "${1:-}" ]; then
		local _input
		for _input in "${@:-}"; do
			case ${_input} in
				-m=* | --module-name=*)
					_MODULE_NAME="${_input#*=}"
					shift;;
				-d=* | --module-dir=*)
					_MODULE_DIR="${_input#*=}"
					shift;;
				-o=* | --output-dir=*)
					_OUTPUT_DIR="${_input#*=}"
					shift;;
				*)
					echo "[ERROR]: Failed to parsing input -> ${_input}!"
					echo "[INFO]: USAGE: ${0}  -m=*, --module-name=* [my_module01] | -d=*, --module-dir=* [./src/my_module01] | -o=*, --output-dir=* [./docs/diagrams]"
					exit 1;;
			esac
		done
	fi
	## --- Menu arguments --- ##


	if [ -z "${_MODULE_NAME:-}" ]; then
		_MODULE_NAME="${MODULE_NAME}"
	else
		MODULE_DIR="./src/${_MODULE_NAME}"
	fi

	if [ -z "${_MODULE_DIR:-}" ]; then
		_MODULE_DIR="${MODULE_DIR}"
	fi

	if [ -z "${_OUTPUT_DIR:-}" ]; then
		_OUTPUT_DIR="${OUTPUT_DIR}"
	fi

	local _classes_dir="${_OUTPUT_DIR}/classes"
	local _packages_dir="${_OUTPUT_DIR}/packages"

	if [ ! -d "${_classes_dir}" ]; then
		mkdir -vp "${_classes_dir}"
	fi

	if [ ! -d "${_packages_dir}" ]; then
		mkdir -vp "${_packages_dir}"
	fi


	echo "[INFO]: Generating UML diagrams..."
	local _cp_formats=("html" "pdf" "png" "svg")
	local _cp_format
	for _cp_format in "${_cp_formats[@]}"; do
		local _tmp_class_path="${_OUTPUT_DIR}/classes_${_MODULE_NAME}.${_cp_format}"
		local _tmp_package_path="${_OUTPUT_DIR}/packages_${_MODULE_NAME}.${_cp_format}"

		echo "[INFO]: Generating ['${_tmp_class_path}', '${_tmp_package_path}'] files..."
		pyreverse -d "${_OUTPUT_DIR}" -o "${_cp_format}" -p "${_MODULE_NAME}" "${_MODULE_DIR}" || exit 2
		mv -vf "${_tmp_class_path}" "${_classes_dir}/" || exit 2
		mv -vf "${_tmp_package_path}" "${_packages_dir}/" || exit 2
		echo "[OK]: Done."
	done
	echo "[OK]: Done."
}

main "${@:-}"
## --- Main --- ##
