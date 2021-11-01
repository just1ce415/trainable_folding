#!/bin/bash

echo scp events* ripper:~/projects/alphadock/data_preparation/data/runs/$(basename $(pwd))
scp events* ripper:~/projects/alphadock/data_preparation/data/runs/$(basename $(pwd))