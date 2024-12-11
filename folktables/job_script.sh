#!/bin/bash
#SBATCH --job-name=multifair-mixup
#SBATCH --output=mfm.out
#SBATCH --error=mfm.err
#SBATCH --partition=long,general,r3lit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

for DS in income employment;
do
	# eo dp mpma mpmc base
	for mode in mpma mpmc;
	do
		echo $DS;
		echo $mode;
		python main.py --mode $mode \
					   --dsname $DS \
					   --state CA \
					   --year 2022
	done
done

bash aggregate_results.sh