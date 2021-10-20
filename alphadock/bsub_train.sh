#!/bin/bash
#BSUB -P BIP215
#BSUB -W 6:00
#BSUB -nnodes 91
#?SUB -q debug
#?SUB -J RunSim123
#?SUB -o RunSim123.%J
#?SUB -e RunSim123.%J

source /ccs/home/ignatovmg/projects/alphadock/activate_summit.sh

wdir=/ccs/home/ignatovmg/projects/alphadock/data_preparation/data/runs/run1
cd $wdir
cp /ccs/home/ignatovmg/projects/alphadock/alphadock/train.py .
jsrun -n $((6 * 91)) -r 6 -a 1 -g 1 -c 4 python /ccs/home/ignatovmg/projects/alphadock/alphadock/train.py train_epoch_6_loss_37.711.pth
#jsrun -n 2 -r 1 -a 1 -g 6 -c 4 python /ccs/home/ignatovmg/projects/alphadock/alphadock/train.py