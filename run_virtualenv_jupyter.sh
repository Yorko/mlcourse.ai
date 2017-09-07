#!/usr/bin/env bash

virtualenv_dir='venv'

log() {
    echo -e '['$(date '+%Y.%m.%d %H:%M:%S')']: '"${*}"
}

[ ! -d "${virtualenv_dir}" ] && { log 'Initializing virtualenv.' && { virtualenv "${virtualenv_dir}" || exit ${?}; } }
log 'Activating virtualenv.' && { source "${virtualenv_dir}/bin/activate" || exit ${?}; }
log 'Installing and upgrading packages in virtualenv.' && { pip install -r requirements.txt || exit ${?}; }

jupyter notebook

log 'Deactivating virtualenv.' && deactivate