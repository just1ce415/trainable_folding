#!/bin/bash

input_dir="/storage/erglukhov/fragments/0000035/sdf_files/"
output_dir="/storage/erglukhov/fragments/0000035/renumbered_sdf_files/"

for file in "$input_dir"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        python3 renumber_sdf.py -f "$file" -a Cl -o "$output_dir/$filename" > logs
    fi
done