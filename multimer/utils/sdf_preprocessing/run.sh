#!/bin/bash

input_dir="/storage/erglukhov/fragments/0000035/sdf_files/"
output_dir="/storage/erglukhov/fragments/0000035/renumbered_sdf_files/"
python_script="/path/to/your/python/script.py"

for file in "$input_dir"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        python3 "$python_script" < "$file" > "$output_dir/$filename"
    fi
done