#!/bin/bash
set -euo pipefail


## --- Base --- ##
# Getting path of this script file:
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_PROJECT_DIR="$(cd "${_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${_PROJECT_DIR}" || exit 2


if [ -z "$(which git)" ]; then
	echo "[ERROR]: 'git' not found or not installed!"
	exit 1
fi

if [ -z "$(which gh)" ]; then
	echo "[ERROR]: 'gh' not found or not installed!"
	exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
	echo "[ERROR]: You need to login: 'gh auth login'!"
	exit 1
fi
## --- Base --- ##


## --- Variables --- ##
# Flags:
_IS_BUILD=false
## --- Variables --- ##


## --- Main --- ##
main()
{
	## --- Menu arguments --- ##
	if [ -n "${1:-}" ]; then
		local _input
		for _input in "${@:-}"; do
			case ${_input} in
				-b | --build)
					_IS_BUILD=true
					shift;;
				*)
					echo "[ERROR]: Failed to parsing input -> ${_input}!"
					echo "[INFO]: USAGE: ${0}  -b, --build"
					exit 1;;
			esac
		done
	fi
	## --- Menu arguments --- ##


	if [ "${_IS_BUILD}" == true ]; then
		./scripts/build.sh -c
	fi

	local _current_version
	_current_version="$(./scripts/get-version.sh)"
	echo "[INFO]: Creating release for version: 'v${_current_version}'..."
	gh release create "v${_current_version}" ./dist/* --generate-notes
	echo "[OK]: Done."
}

main "${@:-}"
## --- Main --- ##
