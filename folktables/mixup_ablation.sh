#!/bin/bash
#SBATCH --job-name=mixup
#SBATCH --output=mixup.out
#SBATCH --error=mixup.err
#SBATCH --partition=long,general,r3lit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

# for year in 2022 2019 2021 2018;
# do
# 	for state in CA TX FL NY PA IL OH GA NC MI;
# 	do
# 		for DS in employment income;
# 		do
# 			echo "$DS base $state $year";
# 			python main.py --mode base \
# 						   --dsname $DS \
# 						   --state $state \
# 						   --year $year \
# 						   --mixup
# 		done
# 	done
# done


# for k in 3 100 1 40;
# do
# 	for lam in 0.5 0.25;
# 	do
# 		for DS in employment income;
# 		do
# 			cp results/$DS/CA/2022/base_mixup/lam_0.5_d_10_alpha_1_niter_100_k_3_all.csv \
# 			   results/$DS/CA/2022/base_mixup/lam_"$lam"_d_10_alpha_1_niter_100_k_"$k"_all.csv
# 			for mode in mpma mpmc eo;
# 			do
# 				echo "$DS, $mode, $lam, $k"
# 				python main.py --mode $mode \
# 							   --dsname $DS \
# 							   --state CA \
# 							   --year 2022 \
# 							   --lam $lam \
# 							   --k $k \
# 							   --mixup  
# 			done
# 		done
# 	done
# done

# for setting in 'dis' 'one' 'all' 'big' 'small' ;
# do
# 	for year in 2022 2019 2021 2018;
# 	do
# 		for state in CA TX FL NY PA IL OH GA NC MI;
# 		do
# 			for DS in income employment;
# 			do
# 				for mode in mpma mpmc eo;
# 				do
# 					echo "$DS, $mode, $state, $year, $setting"
# 					python main.py --mode $mode \
# 							   --dsname $DS \
# 							   --state $state \
# 							   --year $year \
# 							   --setting $setting \
# 							   --mixup --best
# 				done
# 			done	
# 		done
# 	done
# done

# 'one' 'dis' 'small' 'big' 'all'
for setting in 'small' 'big' 'all';
do
	for year in 2019 2018 2021 2022;
	do
		for state in PA NY FL TX CA MI NC GA OH IL;
		do
			for DS in income employment;
			do
				for mode in enforce_mc enforce_ma;
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