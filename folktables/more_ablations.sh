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

for year in 2022 2019 2021 2018;
do
	for state in CA TX FL NY PA IL OH GA NC MI;
	do
		for DS in income employment;
		do
			for mode in mpma mpmc eo dp;
			do
				for setting in 'big' 'small' 'dis' 'one';
				do
					echo "$DS, $mode, $state, $year"
					python main.py --mode $mode \
							   --dsname $DS \
							   --state $state \
							   --year $year \
							   --setting $setting \
							   --best
				done
			done
		done	
	done
	rm -r data/$year
done