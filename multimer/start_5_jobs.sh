#!/bin/bash

for i in {1..5}; do
    echo "Starting summit_job.sh with parameter: $i"

	job_name="job_$i"
	job_script="${job_name}.sh"

	# Read the job_template.lsf and replace the placeholders with actual values
	sed -e "s/{{MODEL_VERSION}}/$i/g" run_olcf.sh > "$job_script"

	# Submit the LSF job script using bsub
	bsub $job_script

	echo "Submitted job" $job_name

	rm $job_script

  sleep 1

done

echo "All jobs submitted."