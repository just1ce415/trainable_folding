#!/bin/bash

files="${@}"
if [ $# -eq 0 ]; then
        files="{*sh,*py}"
fi

echo scp ripper:"/home/ignatovmg/projects/alphadock/$(realpath --relative-to=$ROOT $(pwd))/$files" .
scp ripper:"/home/ignatovmg/projects/alphadock/$(realpath --relative-to=$ROOT $(pwd))/$files" .