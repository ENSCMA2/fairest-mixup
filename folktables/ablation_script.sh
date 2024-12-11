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

# for DS in employment income;
# do
# 	echo "$DS base";
# 	python test.py;
# 	python main.py --mode base \
# 				   --dsname $DS \
# 				   --state CA \
# 				   --year 2022;
# done


# for k in 3 100 1 40;
# do
# 	for lam in 0.5 0.25;
# 	do
# 		for DS in employment income;
# 		do
# 			cp results/$DS/CA/2022/base/lam_0.5_d_10_alpha_1_niter_100_k_3_all.csv \
# 			   results/$DS/CA/2022/base/lam_"$lam"_d_10_alpha_1_niter_100_k_"$k"_all.csv
# 			for mode in mpma eo dp;
# 			do
# 				echo "$DS, $mode, $lam, $d, $k"
# 				python main.py --mode $mode \
# 							   --dsname $DS \
# 							   --state CA \
# 							   --year 2022 \
# 							   --lam $lam \
# 							   --k $k
							   
# 			done
# 			for d in 100 55 10;
# 			do
# 				echo "$DS, mpmc, $lam, $d, $k"
# 				python main.py --mode mpmc \
# 							   --dsname $DS \
# 							   --state CA \
# 							   --year 2022 \
# 							   --lam $lam \
# 							   --d $d \
# 							   --k $k
# 				for mode in mpma eo dp base;
# 				do
# 					cp results/$DS/CA/2022/$mode/lam_"$lam"_d_10_alpha_1_niter_100_k_3_all.csv \
# 					   results/$DS/CA/2022/$mode/lam_"$lam"_d_"$d"_alpha_1_niter_100_k_"$k"_all.csv
					   
# 				done
# 				python aggregate_results.py --ds $DS --state CA \
# 										--year 2022 \
# 										--lam $lam --d $d --k $k
# 			done
# 		done
# 	done
# done

# for setting in 'big' 'small' 'dis' 'one';
# do
# 	for DS in income employment;
# 	do
# 		for mode in mpmc mpma eo dp;
# 		do
# 			echo "$DS, $mode, $setting";
# 			python main.py --mode $mode \
# 						   --dsname $DS \
# 						   --state CA \
# 						   --year 2022 \
# 						   --setting $setting \
# 						   --best
# 		done
# 	done
# done


# # https://www.statista.com/statistics/183497/population-in-the-federal-states-of-the-us/
for year in 2022 2019 2021 2018 2020;
do
	for state in CA TX FL NY PA IL OH GA NC MI;
	do
		for DS in income employment;
		do
			for mode in mpma mpmc eo dp base;
			do
				echo "$DS, $mode, $state, $year"
				python main.py --mode $mode \
							   --dsname $DS \
							   --state $state \
							   --year $year \
							   --best
			done
		done	
	done
	rm -r data/$year
done
