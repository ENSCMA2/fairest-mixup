#!/bin/bash
#SBATCH --job-name=fairbase
#SBATCH --output=fairbase.out
#SBATCH --error=fairbase.err
#SBATCH --partition=long,general,r3lit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00

for year in 2022 2019 2021 2018;
do
	for state in CA TX FL NY PA IL OH GA NC MI;
	do
		for DS in income employment;
		do
			for mode in mpma mpmc eo;
			do
				for setting in 'all' 'big' 'small' 'dis' 'one';
				do
					echo "$DS, $mode, $state, $year, $setting"
					python main.py --mode $mode \
							   --dsname $DS \
							   --state $state \
							   --year $year \
							   --lam 0 \
							   --setting $setting
				done
			done
		done	
	done
done