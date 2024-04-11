#!/bin/bash

# Get the directory from input
dir_path="$1"

# Ensure the directory path ends with a slash
if [[ "${dir_path: -1}" != "/" ]]; then
    dir_path="${dir_path}/"
fi

# add pwd to pythonpath
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python "${dir_path}zero_to_fp32.py" "${dir_path}" "${dir_path}pytorch_model.bin"
cp "${dir_path}../config.yaml" "${dir_path}/config.yaml"
python "./trainer/scripts/convert_weights_to_hf.py" "--config-path" "${dir_path}" "output_dir=${dir_path}"
