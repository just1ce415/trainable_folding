#!/bin/bash
#$ -P k3domics
#$ -l h_rt=11:00:00   # Specify the hard time limit for the job
#$ -pe omp 28
#$ -j y               # Merge the error and output streams into a single file
#$ -t 1-512
#$ -o logs/

set -euo pipefail

NCPU=28
HHLIB=/projectnb/k3domics/ignatovm/software/hh-suite/build
#UNIPROT=/data/databases/hhpred/uniclust30_2018_08/uniclust30_2018_08
PDB100=/projectnb2/sc3dm/ignatovm/databases/hhpred/pdb100_2020_03_15/pdb100_prot

export HHLIB

#SGE_TASK_ID=2
#SGE_TASK_LAST=512

task_id=$((SGE_TASK_ID-1))
num_tasks=$SGE_TASK_LAST
counter=-1

cd /projectnb/sc3dm/ignatovm/projects/alphadock/data_preparation/data/hhpred 
echo task_id: $task_id
echo num_tasks: $num_tasks

cat ../chains.txt | while read line
do
  # somehow if i write ((counter++)), it interacts with "set -e" flag and exits
  counter=$((counter+1))
  id=$line
  if [ $task_id -ne $(($counter % $num_tasks)) ]; then
    continue
  fi
  echo ">>>>> Processing $id [$counter] <<<<<"
  if [ ! -f "$id.a3m" ]; then
      #"$HHLIB/bin/hhblits" -maxfilt 100000 -realign_max 100000 -d "$UNIPROT" -all -B 100000 -Z 100000 -n 3 -e 0.001 -cpu $NCPU -i "$i" -oa3m "$id.a3m" -o "${id}-uniprot.hhr"
      "$HHLIB/bin/ffindex_get" "${PDB100}_a3m.ffdata" "${PDB100}_a3m.ffindex" "${id}" > "$id.a3m"
  fi
  if [ ! -f "$id.hhr" ]; then
      "$HHLIB/bin/hhsearch" -d "$PDB100" -p 20 -Z 50000 -loc -z 1 -b 1 -B 50000 -ssm 2 -sc 1 -seq 1 -dbstrlen 10000 -norealign -maxres 32000 -cpu $NCPU -i "$id.a3m" -o "$id.hhr"
  fi
done
