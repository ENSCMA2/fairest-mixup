#!/bin/bash
#SBATCH --job-name=enforce
#SBATCH --output=enforce.out
#SBATCH --error=enforce.err
#SBATCH --partition=long,general,r3lit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

for setting in 'all' 'big' 'small';
do
	for year in 2022 2019 2021 2018;
	do
		for state in CA TX FL NY PA IL OH GA NC MI;
		do
			for DS in income employment;
			do
				for mode in enforce_ma enforce_mc;
				do
					echo "$DS, $mode, $state, $year, $setting"
					python main.py --mode $mode \
							   --dsname $DS \
							   --state $state \
							   --year $year \
							   --setting $setting
				done
			done	
		done
	done
done